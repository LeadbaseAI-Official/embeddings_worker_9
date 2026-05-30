import os
import sys
import subprocess
from typing import List
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

TEMP_TOKENS_BIN = "combined_tokens.bin"
TEMP_COUNTS_BIN = "counts_temp.bin"

def get_last_processed_shard_index(api: HfApi) -> int:
    """
    Scan the output repository to find files matching 'counts_{WORKER_ID}_s{SHARD_INDEX}.bin'.
    Extract and return the maximum SHARD_INDEX found.
    """
    print(f"Scanning output repository {HF_OUTPUT_REPO} for worker {WORKER_ID} progress...", flush=True)
    try:
        all_files: List[str] = api.list_repo_files(repo_id=HF_OUTPUT_REPO, repo_type="dataset", token=HF_TOKEN)
    except Exception as e:
        print(f"Error listing output repo files: {e}. Assuming fresh start.", flush=True)
        return -1

    prefix = f"counts_{WORKER_ID}_s"
    suffix = ".bin"
    last_idx = -1
    
    for filename in all_files:
        if filename.startswith(prefix) and filename.endswith(suffix):
            try:
                # Extract index from counts_W_s{INDEX}.bin
                idx_str = filename[len(prefix): -len(suffix)]
                idx = int(idx_str)
                if idx > last_idx:
                    last_idx = idx
            except ValueError:
                continue

    print(f"Highest processed shard index found: {last_idx}", flush=True)
    return last_idx

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

    import shutil

    api = HfApi()

    # 1. Fetch the last completed shard index from output file naming
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

    accumulated_file = "accumulated_counts.bin"
    if os.path.exists(accumulated_file):
        os.remove(accumulated_file)

    # If we resumed from a previously completed index, download the latest counts file
    if last_completed_idx >= 0:
        prev_filename = f"counts_{WORKER_ID}_s{last_completed_idx}.bin"
        print(f"Downloading previous counts file {prev_filename} to resume...", flush=True)
        try:
            downloaded = hf_hub_download(
                repo_id=HF_OUTPUT_REPO,
                filename=prev_filename,
                repo_type="dataset",
                token=HF_TOKEN
            )
            shutil.copy(downloaded, accumulated_file)
            print(f"Resumed successfully from shard index {last_completed_idx}.", flush=True)
        except Exception as e:
            print(f"Warning: Could not download previous progress file: {e}. Starting fresh.", flush=True)
            last_completed_idx = -1

    batch_size = MAX_SHARDS_PER_RUN
    current_idx = 0
    max_completed_idx = last_completed_idx

    while current_idx < len(shards_to_process):
        batch = shards_to_process[current_idx : current_idx + batch_size]
        current_idx += batch_size

        print(f"\n--- Processing batch: {batch} ---", flush=True)

        if os.path.exists(TEMP_TOKENS_BIN):
            os.remove(TEMP_TOKENS_BIN)

        downloaded_parquet_files = []
        try:
            with open(TEMP_TOKENS_BIN, "wb") as f_out:
                for shard_path in batch:
                    idx = extract_shard_index(shard_path)
                    print(f"Downloading and extracting: {shard_path} (Index: {idx})...", flush=True)

                    local_parquet = hf_hub_download(
                        repo_id=HF_INPUT_REPO,
                        filename=shard_path,
                        repo_type="dataset",
                        token=HF_TOKEN
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

            # Merge iteration counts into accumulated counts
            new_accumulated = "accumulated_counts_new.bin"
            if os.path.exists(new_accumulated):
                os.remove(new_accumulated)

            print(f"Merging iteration counts into accumulated counts...", flush=True)
            subprocess.run([binary_name, "--merge", accumulated_file, TEMP_COUNTS_BIN, new_accumulated], check=True)

            if os.path.exists(accumulated_file):
                os.remove(accumulated_file)
            os.rename(new_accumulated, accumulated_file)

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

    # 3. Upload the final co-occurrence binary file at the very end
    if os.path.exists(accumulated_file):
        dest_filename = f"counts_{WORKER_ID}_s{max_completed_idx}.bin"
        print(f"Uploading final output file {accumulated_file} as {dest_filename}...", flush=True)
        api.upload_file(
            path_or_fileobj=accumulated_file,
            path_in_repo=dest_filename,
            repo_id=HF_OUTPUT_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"Upload successful. Completed up to shard index: {max_completed_idx}", flush=True)
        os.remove(accumulated_file)
    else:
        print("Error: accumulated counts file not found at the end of the run.", file=sys.stderr, flush=True)

if __name__ == "__main__":
    run_pipeline()
