import modal

# -------------------------------
# Build the container image
# -------------------------------
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("vllm", "transformers", "torch", "huggingface_hub")
)

# -------------------------------
# Modal App definition
# -------------------------------
app = modal.App(
    name="llama3-vllm-rag",
    image=vllm_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)

# -------------------------------
# Persistent vLLM GPU class
# -------------------------------
@app.cls(
    gpu="A100",
    timeout=600,
    min_containers=1,
    scaledown_window=600,
)
class LLMRunner:
    @modal.enter()
    def load(self):
        from vllm import LLM, SamplingParams

        print("🚀 Loading model...")
        self.llm = LLM(
            model="tomc30098/llama3-8b-qlora-merged",
            tokenizer="tomc30098/llama3-8b-qlora-merged",
            trust_remote_code=True,
            dtype="float16",
        )
        self.params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=256
        )
        print("✅ Model loaded.")

    @modal.method()
    def generate(self, prompt: str) -> str:
        result = self.llm.generate(prompt, self.params)
        return result[0].outputs[0].text

# -------------------------------
# Local test entrypoint
# -------------------------------
@app.local_entrypoint()
def main():
    prompt = "My name is Kai, nice to meet you, Let's get started!!"
    print("🧠 Sending prompt to remote GPU...")

    output = LLMRunner().generate.remote(prompt)
    print("\n💬 Response:\n", output)