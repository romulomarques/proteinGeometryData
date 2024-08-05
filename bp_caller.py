import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

os.makedirs("xbsol_leftmost", exist_ok=True)

def call_app(file_path):
    try:
        result = subprocess.run(['./bp.exe', file_path], timeout=60, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return None, "Timeout", ""

def main(folder_path):
    files = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            files.append(os.path.join(folder_path, file))

    max_workers = os.cpu_count() - 1
    return_codes = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(call_app, file) for file in files]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            return_code, _, _ = future.result()
            return_codes.append(return_code)

    
    print(f"Total files processed: {len(files)}")
    print(f"Total files successfully processed: {return_codes.count(0)}")
    print(f"Total files failed: {return_codes.count(1)}")
    print(f"Total files timed out: {return_codes.count(None)}")
    

if __name__ == "__main__":
    folder_path = 'dmdgp'  # Replace with your folder path
    main(folder_path)
