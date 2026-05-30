import os
import sys
import subprocess
from typing import List
import json
import urllib.request
import time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import numpy as np
from huggingface_hub import HfApi, hf_hub_download

# ============================================================
# CONFIGURATION (WORKER_ID is patched during deployment)
# ============================================================
WORKER_ID = 9

HF_INPUT_REPO = "anisoleai/fineweb-tokenized"
HF_OUTPUT_REPO = "anisoleai/embeddings"
HF_TOKEN = os.getenv("HF_TOKEN")
WINDOW_SIZE = 5
MAX_SHARDS_PER_RUN = 2
GITHUB_ORG = "LeadbaseAI-Official"

TEMP_TOKENS_BIN = "combined_tokens.bin"
TEMP_COUNTS_BIN = "counts_temp.bin"

def get_last_processed_shard_index(api: HfApi) -> int:
    """
    Download and parse metadata.json from HF output repository for worker progress.
    """
    metadata_path = f"data_{WORKER_ID}/metadata.json"
    print(f"Downloading metadata file {metadata_path} from HF...", flush=True)
    try:
        local_path = hf_hub_download(
            repo_id=HF_OUTPUT_REPO,
            filename=metadata_path,
            repo_type="dataset",
            token=HF_TOKEN
        )
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            last_idx = data.get("last_processed_shard_index", -1)
            print(f"Highest processed shard index from metadata: {last_idx}", flush=True)
            return last_idx
    except Exception as e:
        print(f"Error downloading/reading metadata file: {e}. Assuming fresh start (-1).", flush=True)
        return -1

def save_metadata(api: HfApi, last_idx: int) -> None:
    """
    Write and upload updated metadata.json to HF output repository.
    """
    metadata_path = f"data_{WORKER_ID}/metadata.json"
    local_path = "metadata.json"
    try:
        data = {"last_processed_shard_index": last_idx}
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        print(f"Uploading updated metadata to {metadata_path} (index: {last_idx})...", flush=True)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=metadata_path,
            repo_id=HF_OUTPUT_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
    except Exception as e:
        print(f"Error saving/uploading metadata: {e}", flush=True)
    finally:
        if os.path.exists(local_path):
            os.remove(local_path)

# Chain triggering is disabled.

def get_input_shards_list(api: HfApi) -> List[str]:
    """
    List all parquet shards under data_{WORKER_ID}/ directory in the input Hugging Face repository.
    """
    print(f"Listing parquet shards in {HF_INPUT_REPO} under data_{WORKER_ID}/...", flush=True)
    all_files: List[str] = api.list_repo_files(repo_id=HF_INPUT_REPO, repo_type="dataset", token=HF_TOKEN)
    
    folder_prefix = f"data_{WORKER_ID}/"
    shard_files: List[str] = [
        f for f in all_files 
        if f.startswith(folder_prefix) and f.endswith(".parquet")
    ]
    
    # Sort files naturally by name to process them sequentially
    shard_files.sort()
    print(f"Found {len(shard_files)} shards for worker {WORKER_ID}.", flush=True)
    return shard_files

def extract_shard_index(filename: str) -> int:
    """
    Helper to extract the numeric shard index from parquet paths like data_1/shard-00005.parquet
    """
    basename = os.path.basename(filename)
    digits = "".join([c for c in basename if c.isdigit()])
    return int(digits) if digits else 0

def run_pipeline() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN is not set in environment.", file=sys.stderr, flush=True)
        sys.exit(1)

    api = HfApi()

    # 1. Fetch progress from metadata.json
    last_completed_idx = get_last_processed_shard_index(api)

    # 2. Fetch input shards list
    all_shards = get_input_shards_list(api)
    
    # Filter for shards that have not been processed yet
    shards_to_process: List[str] = []
    for path in all_shards:
        idx = extract_shard_index(path)
        if idx > last_completed_idx:
            shards_to_process.append(path)

    if not shards_to_process:
        print(f"All shards for Worker {WORKER_ID} are already processed! Exiting.", flush=True)
        return

    print(f"Worker {WORKER_ID} has {len(shards_to_process)} shards remaining to process.", flush=True)

    # Compile comat.cpp if binary is missing
    binary_name = "./comat" if not sys.platform.startswith("win") else "comat.exe"
    if not os.path.exists(binary_name):
        print(f"Compiling comat.cpp...", flush=True)
        cmd = ["g++", "-O3", "-march=native", "-o", binary_name, "comat.cpp"]
        subprocess.run(cmd, check=True)
        print("Compilation successful.", flush=True)

    # Capped at 4 shards per runner run
    SHARDS_LIMIT_THIS_RUN = 4
    batch_size = MAX_SHARDS_PER_RUN
    
    # We only process up to SHARDS_LIMIT_THIS_RUN
    run_shards = shards_to_process[:SHARDS_LIMIT_THIS_RUN]
    print(f"Worker {WORKER_ID} will process {len(run_shards)} shards in this execution.", flush=True)

    current_idx = 0
    max_completed_idx = last_completed_idx

    while current_idx < len(run_shards):
        batch = run_shards[current_idx : current_idx + batch_size]
        current_idx += batch_size

        print(f"\n--- Processing batch: {batch} ---", flush=True)

        if os.path.exists(TEMP_TOKENS_BIN):
            os.remove(TEMP_TOKENS_BIN)

        downloaded_parquet_files = []
        try:
            batch_start_idx = extract_shard_index(batch[0])
            batch_end_idx = extract_shard_index(batch[-1])

            with open(TEMP_TOKENS_BIN, "wb") as f_out:
                for shard_path in batch:
                    idx = extract_shard_index(shard_path)
                    print(f"Downloading and extracting: {shard_path} (Index: {idx})...", flush=True)

                    local_parquet = hf_hub_download(
                        repo_id=HF_INPUT_REPO,
                        filename=shard_path,
                        repo_type="dataset",
                        token=HF_TOKEN,
                        local_dir=".",
                        local_dir_use_symlinks=False
                    )
                    downloaded_parquet_files.append(local_parquet)

                    table = pq.read_table(local_parquet, columns=["token_ids"])
                    tokens_np = table["token_ids"].to_numpy(zero_copy_only=False)
                    f_out.write(tokens_np.tobytes())
                    max_completed_idx = max(max_completed_idx, idx)

            if os.path.exists(TEMP_COUNTS_BIN):
                os.remove(TEMP_COUNTS_BIN)

            print(f"Starting C++ counting on batch tokens...", flush=True)
            subprocess.run([binary_name, TEMP_TOKENS_BIN, TEMP_COUNTS_BIN, str(WINDOW_SIZE)], check=True)
            print("C++ counting completed successfully.", flush=True)

            # Upload counts binary file directly to HF (no local merging!)
            timestamp = int(time.time())
            dest_filename = f"data_{WORKER_ID}/counts_s{batch_start_idx}_s{batch_end_idx}_{timestamp}.bin"
            print(f"Uploading batch output file {TEMP_COUNTS_BIN} as {dest_filename}...", flush=True)
            api.upload_file(
                path_or_fileobj=TEMP_COUNTS_BIN,
                path_in_repo=dest_filename,
                repo_id=HF_OUTPUT_REPO,
                repo_type="dataset",
                token=HF_TOKEN
            )
            print(f"Upload successful. Completed batch shards: s{batch_start_idx} to s{batch_end_idx}", flush=True)

            # Save progress in metadata
            save_metadata(api, max_completed_idx)

        finally:
            # Cleanup temporary files
            for temp_file in [TEMP_TOKENS_BIN, TEMP_COUNTS_BIN]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            # Cleanup downloaded parquet files
            for pq_file in downloaded_parquet_files:
                if os.path.exists(pq_file):
                    try:
                        os.remove(pq_file)
                    except Exception as e:
                        print(f"Warning: Failed to delete local parquet file {pq_file}: {e}", flush=True)

    # Check if there are still shards remaining in total queue
    remaining_shards = shards_to_process[len(run_shards):]
    if remaining_shards:
        print(f"Workflow completed processing {len(run_shards)} shards.", flush=True)
        print(f"{len(remaining_shards)} shards remaining in total queue.", flush=True)
    else:
        print("All remaining shards have been successfully processed!", flush=True)

if __name__ == "__main__":
    run_pipeline()
