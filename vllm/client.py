import modal

stub = modal.App.lookup("llama3-vllm-rag")
generate = stub.function("generate")
response = generate.remote("Tell me about the Faroe Islands.")
print(response)
