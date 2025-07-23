# 使用官方 vLLM OpenAI 伺服器映像，內含 CUDA 12.9 支援
FROM vllm/vllm-openai:latest

# 設定工作目錄
WORKDIR /workspace

# 複製專案檔案到容器內（你所有的 Python 檔、.env、LoRA 等）
COPY . .

RUN pip install --no-cache-dir -r vllm/requirements.txt

# 預設執行指令（可以是你的推論腳本）
ENTRYPOINT ["python3", "vllm/vllm_inference.py"]
