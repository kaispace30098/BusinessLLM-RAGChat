import modal

# Build the container image with vLLM + HF Hub support
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm", "transformers", "torch", "huggingface_hub")
)

# Define the Modal App and inject your HF_TOKEN secret
app = modal.App(
    "llama3-vllm-rag",
    image=vllm_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

# GPU function definition
@app.function(
    gpu="A10G",
    timeout=600,
    min_containers=1,
    scaledown_window=180,
)
async def generate(prompt: str) -> str:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="tomc30098/llama3-8b-qlora-merged",
        tokenizer="tomc30098/llama3-8b-qlora-merged",
        trust_remote_code=True,
        dtype="float16",
    )
    params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)
    result = llm.generate(prompt, params)
    return result[0].outputs[0].text


# Local entrypoint: skip user input, send hardcoded prompt
@app.local_entrypoint()
def main():
    prompt = "My name is Kai, nice to meet you, Let's get started!!"
    print("Sending prompt to remote GPU...")
    output = generate.remote(prompt)
    print("\nResponse:\n", output)

