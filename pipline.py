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

    # Cap to max 60 shards per run to comfortably process within Actions runtime limits
    max_shards_per_run = 60
    shards_to_process = shards_to_process[:max_shards_per_run]
    print(f"Worker {WORKER_ID} will process {len(shards_to_process)} shards in this run.", flush=True)

    # 3. Concatenate all shard tokens into a single binary file
    if os.path.exists(TEMP_TOKENS_BIN):
        os.remove(TEMP_TOKENS_BIN)

    max_completed_idx = last_completed_idx

    try:
        with open(TEMP_TOKENS_BIN, "wb") as f_out:
            for shard_path in shards_to_process:
                idx = extract_shard_index(shard_path)
                print(f"Downloading and extracting: {shard_path} (Index: {idx})...", flush=True)
                
                # Download parquet shard locally
                local_parquet = hf_hub_download(
                    repo_id=HF_INPUT_REPO,
                    filename=shard_path,
                    repo_type="dataset",
                    token=HF_TOKEN
                )
                
                # Load parquet table and extract token_ids column
                table = pq.read_table(local_parquet, columns=["token_ids"])
                
                # Flatten the Arrow list structure natively to keep RAM under 200MB (prevents OOM crashes)
                flat_tokens = pc.cast(table["token_ids"].flatten(), pa.uint16())
                tokens_np = flat_tokens.to_numpy(zero_copy_only=False)
                
                # Append raw uint16 bytes directly to combined file
                f_out.write(tokens_np.tobytes())
                
                # Track the highest index successfully compiled
                max_completed_idx = max(max_completed_idx, idx)

        # 4. Compile comat.cpp if binary is missing
        binary_name = "./comat" if not sys.platform.startswith("win") else "comat.exe"
        if not os.path.exists(binary_name):
            print(f"Compiling comat.cpp...", flush=True)
            cmd = ["g++", "-O3", "-march=native", "-o", binary_name, "comat.cpp"]
            subprocess.run(cmd, check=True)
            print("Compilation successful.", flush=True)

        # 5. Run the C++ co-occurrence counter directly on the combined binary file
        print(f"Starting C++ counting on {TEMP_TOKENS_BIN}...", flush=True)
        if os.path.exists(TEMP_COUNTS_BIN):
            os.remove(TEMP_COUNTS_BIN)

        # Run C++ process (which displays the C++ native progress bar on stderr)
        subprocess.run([binary_name, TEMP_TOKENS_BIN, TEMP_COUNTS_BIN, str(WINDOW_SIZE)], check=True)
        print("C++ counting completed successfully.", flush=True)

        # 6. Upload final co-occurrence binary file
        dest_filename = f"counts_{WORKER_ID}_s{max_completed_idx}.bin"
        print(f"Uploading output file {TEMP_COUNTS_BIN} as {dest_filename}...", flush=True)
        api.upload_file(
            path_or_fileobj=TEMP_COUNTS_BIN,
            path_in_repo=dest_filename,
            repo_id=HF_OUTPUT_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        print(f"Upload successful. Completed up to shard index: {max_completed_idx}", flush=True)

    finally:
        # Clean up temporary files to release disk space
        for temp_file in [TEMP_TOKENS_BIN, TEMP_COUNTS_BIN]:
            if os.path.exists(temp_file):
                print(f"Cleaning up local file: {temp_file}", flush=True)
                os.remove(temp_file)
        print("Cleanup done.", flush=True)

if __name__ == "__main__":
    run_pipeline()
