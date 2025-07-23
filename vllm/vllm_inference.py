# vllm_inference.py
# Script for inference using vLLM engine only
# It prints the input prompt, the generated response, and the inference time.

import os
import credential  # Load environment variables from .env
import time
from vllm import LLM

# Retrieve Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
print("[INFO] HF_TOKEN loaded?", HF_TOKEN is not None)

# Hardcoded instruction prompt for tweet composition
sample = (
    "Instruction: Compose a tweet according to the given facts.\n"
    "Input: Hashtag: #TakeCare\n"
    "Message: Make sure you take time for yourself today.\n"
    "Response:"
)
print("\n[INPUT SAMPLE]\n", sample)

# Initialize vLLM client with corrected dtype
client = LLM(
    model="/workspace/merged_llama3",  
    dtype="float16"
)


# Perform inference and measure time
start_time = time.perf_counter()
out = client.generate([sample], max_tokens=128)
end_time = time.perf_counter()

# Extract and print response
response = out[0].text.strip()
print("\n[vLLM OUTPUT]\n", response)
print(f"[vLLM INFERENCE TIME] {end_time - start_time:.3f} seconds")
