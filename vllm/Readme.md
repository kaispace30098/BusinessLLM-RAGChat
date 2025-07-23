docker build -t my-vllm-lora -f vllm/Dockerfile .
docker run --gpus all --env-file vllm/.env -v ${env:USERPROFILE}\.cache\huggingface:/root/.cache/huggingface -v ${PWD}\merged_llama3:/workspace/merged_llama3 my-vllm-lora
