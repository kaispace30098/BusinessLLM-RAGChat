import modal

def main():
    # Lookup the remote class
    LLMRunner = modal.Cls.from_name("llama3-vllm-rag", "LLMRunner")

    # Create an instance (on GPU)
    llm = LLMRunner()

    # Call the `.generate` method remotely
    result = llm.generate.remote("Tell me a joke about puffins in the Faroe Islands.")

    print("🦜 LLM says:", result)

if __name__ == "__main__":
    main()
