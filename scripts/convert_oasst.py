import gzip
import json
import requests
import boto3
import random
from pathlib import Path

# ---------- CONFIGURATION ----------
BUCKET_NAME = "my-qa-dataset"        # Replace with your S3 bucket name
PREFIX      = "data/"                # Replace with your desired S3 key prefix
ALPACA_URL  = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
OASST1_URL  = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/2023-04-12_oasst_ready.trees.jsonl.gz"
# -------------------------------

# Determine the script's directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Create a .gitignore in the scripts folder to ignore JSON files if it doesn't exist
gitignore_file = SCRIPT_DIR / '.gitignore'
if not gitignore_file.exists():
    gitignore_file.write_text("*.json\n")

# Helper to get full path inside the scripts folder
def get_local_path(filename: str) -> Path:
    return SCRIPT_DIR / filename


def download_file(url: str, output_name: str):
    """
    Download a file from a URL and save it to the scripts directory.
    """
    dest = get_local_path(output_name)
    print(f"Downloading {url} -> {dest}")
    response = requests.get(url)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved to {dest}")


def extract_gzip(gzip_name: str, output_name: str):
    """
    Extract a .gz file to a text file in the scripts directory.
    """
    src = get_local_path(gzip_name)
    dest = get_local_path(output_name)
    print(f"Extracting {src} -> {dest}")
    with gzip.open(src, 'rt', encoding='utf-8') as infile, \
         open(dest, 'w', encoding='utf-8') as outfile:
        for line in infile:
            outfile.write(line)
    print(f"Extracted to {dest}")


def parse_openassistant(input_name: str):
    """
    Parse the OpenAssistant dataset, extracting EN prompter-assistant pairs.
    """
    path = get_local_path(input_name)
    qa_pairs = []
    print(f"Parsing OpenAssistant from {path}")
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            prompt = record.get('prompt', {})
            if prompt.get('role') == 'prompter' and prompt.get('lang') == 'en':
                for reply in prompt.get('replies', []):
                    if reply.get('role') == 'assistant' and reply.get('lang') == 'en':
                        qa_pairs.append({
                            'text': (
                                f"Instruction: {prompt['text'].strip()}\n"
                                f"Response:    {reply['text'].strip()}"
                            )
                        })
                        break
    print(f"Extracted {len(qa_pairs)} QA pairs from OpenAssistant")
    return qa_pairs


def parse_alpaca(input_name: str):
    """
    Parse the Alpaca dataset JSON into QA pairs.
    """
    path = get_local_path(input_name)
    data = json.loads(path.read_text(encoding='utf-8'))
    qa_pairs = []
    print(f"Parsing Alpaca from {path}")
    for entry in data:
        instruction = entry.get('instruction', '').strip()
        input_field = entry.get('input', '').strip()
        output = entry.get('output', '').strip()
        full_instruction = f"{instruction}\nInput: {input_field}" if input_field else instruction
        qa_pairs.append({
            'text': f"Instruction: {full_instruction}\nResponse:    {output}"
        })
    print(f"Extracted {len(qa_pairs)} QA pairs from Alpaca")
    return qa_pairs


def save_jsonl(data: list, filename: str):
    """
    Save a list of dicts to a JSONL file in the scripts directory.
    """
    path = get_local_path(filename)
    with open(path, 'w', encoding='utf-8') as f:
        for record in data:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')
    print(f"Saved {len(data)} records to {path}")


def split_dataset(data: list, train_name: str, eval_name: str, train_frac: float = 0.8):
    """
    Shuffle and split data into training and evaluation sets, then save.
    """
    random.shuffle(data)
    split_index = int(len(data) * train_frac)
    save_jsonl(data[:split_index], train_name)
    save_jsonl(data[split_index:], eval_name)


def upload_files_to_s3(filenames: list, bucket: str, prefix: str):
    """
    Upload specified files from scripts directory to S3.
    """
    s3 = boto3.client('s3')
    for name in filenames:
        local_file = get_local_path(name)
        s3_key = prefix + local_file.name
        print(f"Uploading {local_file} to s3://{bucket}/{s3_key}")
        s3.upload_file(str(local_file), bucket, s3_key)
    print("Uploaded all files to S3")


def main():
    # Step 1: Download and extract OpenAssistant
    download_file(OASST1_URL, 'oasst1.jsonl.gz')
    extract_gzip('oasst1.jsonl.gz', 'oasst1_raw.jsonl')

    # Step 2: Download Alpaca
    download_file(ALPACA_URL, 'alpaca_data.json')

    # Step 3: Parse datasets
    oasst_data = parse_openassistant('oasst1_raw.jsonl')
    alpaca_data = parse_alpaca('alpaca_data.json')

    # Step 4: Combine and split
    combined = oasst_data + alpaca_data
    print(f"Total combined QA pairs: {len(combined)}")
    split_dataset(combined, 'train.jsonl', 'eval.jsonl')

    # Step 5: Upload to S3
    upload_files_to_s3(['train.jsonl', 'eval.jsonl'], BUCKET_NAME, PREFIX)

    # Step 6: Cleanup intermediate files
    for tmp in ['oasst1.jsonl.gz', 'oasst1_raw.jsonl', 'alpaca_data.json']:
        try:
            get_local_path(tmp).unlink()
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    main()
