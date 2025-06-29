# **BusinessLLMRAGChat**

A full-stack, production-grade LLM application that fine-tunes open-source conversational models and deploys them for efficient Retrieval-Augmented Generation (RAG) with LangGraph and vLLM — packaged with Streamlit Cloud UI for business demo.

---

## 🚀 Project Overview

**BusinessLLMRAGChat** is an end-to-end project combining **LLM fine-tuning**, **evaluation**, **accelerated inference**, and **RAG chat demo** to showcase how fine-tuned models can be integrated with external knowledge for business rule reasoning and QA.

This project simulates a real-world pipeline for fine-tuning open-source LLMs (LoRA/SFT), evaluating performance (BLEU, cosine, judge), comparing multiple runs (MLflow), and deploying the best-performing model using `FastAPI + vLLM` for inference.

It ends with a LangGraph-powered **Streamlit Chat App** that supports business-rule-aware question answering and semantic search (via FAISS).

---

## 🧠 Architecture

```
+----------------+       +----------------+       +------------------+       +------------------+
|                |       |                |       |                  |       |                  |
|   User Input   +------>+  Streamlit UI  +------>+     FastAPI      +------>+      vLLM        |
|                |       |  (LangGraph)   |       |   (EC2 Inference) |       | (Accelerated LLM)|
+----------------+       +----------------+       +------------------+       +------------------+
                                                   |
                          +------------------------+-------------------------+
                          |
                          |     Retrieval
                          v
                  [FAISS Vector DB on business rule corpus]
```

---

## 📦 Techniques Used

### 📊 **1. Dataset & Preprocessing**
- OpenAssistant Conversations Dataset (OASST1)
- Converted to `{"text": "Instruction: ...\nResponse: ..."}` format
- Stored in S3 (`train.jsonl`, `eval.jsonl`)

### 🔧 **2. Fine-Tuning Techniques**
- LoRA (Low-Rank Adaptation)
- SFT (Supervised Fine-Tuning)
- Unsloth: Speed-optimized PEFT trainer
- Hugging Face `transformers` + `peft` + `datasets`

### 📈 **3. Experiment Tracking**
- MLflow: Monitor runs, compare BLEU, cosine similarity
- Metrics logging + artifact versioning
- Multiple runs: compare LoRA vs. SFT

### 🏎️ **4. Inference Engine**
- **vLLM**: Token streaming + KV cache optimization
- FastAPI for RESTful inference interface
- Dockerized microservices (FastAPI + vLLM split or combined on EC2 G5)

### 🔍 **5. RAG (Retrieval-Augmented Generation)**
- FAISS similarity search over business rule documents
- LangGraph handles prompt routing and similarity response
- Streamlit connects to FastAPI and triggers semantic retrieval

### 🎯 **6. Deployment**
- Streamlit Cloud frontend demo (LangGraph + chat history)
- EC2 (G5) for both inference containers
- Docker + Poetry for environment reproducibility

---

## 🗂 Directory Structure

```bash
BusinessLLMRAGChat/
├── pyproject.toml                 # Poetry config
├── README.md
├── scripts/
│   └── convert_oasst.py          # Download and convert dataset
├── src/businessllmragchat/
│   ├── fine_tune/                # train.py, evaluate.py
│   ├── rag_langgraph/            # LangGraph RAG pipeline
│   └── inference_api/            # serve_vllm.py, fastapi_app.py
├── inference_api/
│   ├── Dockerfile.vllm
│   ├── Dockerfile.fastapi
│   ├── requirements_vllm.txt
│   └── requirements_api.txt
├── streamlit_demo/
│   ├── app.py
│   └── requirements.txt
└── tests/
    ├── test_train.py
    └── test_inference.py
```

---

## 🛠️ Requirements

- Python 3.10+
- Poetry
- EC2 G5.2xlarge (24GB VRAM) or higher
- Docker
- Hugging Face account (for model downloading)
- AWS IAM role with:
  - `AmazonS3FullAccess`
  - `bedrock:InvokeModel` (optional, if using Bedrock baseline)

---

## 🧪 Evaluation Metrics

| Metric    | Description                                      |
|-----------|--------------------------------------------------|
| **BLEU**  | N-gram overlap between predicted & reference     |
| **Cosine**| Embedding similarity (sentence-transformers)     |
| **Judge** | LLM-based preference evaluation (optional)       |

---

## 📦 Deployment Plan

1. 🎓 Train models using `train.py`
2. 🧪 Evaluate & log runs with MLflow
3. 🥇 Pick best run → serve with FastAPI or vLLM
4. 🚀 Deploy UI via Streamlit Cloud
5. 🔁 LangGraph orchestrates retrieval + inference

---

## 💼 Why This Project Matters

This project simulates a real-world **AI/ML Engineer** workflow:
- Model fine-tuning
- Multi-run comparison & selection
- Inference optimization (vLLM)
- Integration with LangGraph RAG
- Full-stack deployment (backend + frontend)

Ideal as a **portfolio project** to showcase:
- LLM customization
- Infra deployment (EC2, Docker)
- RAG integration for business use cases
- DevOps-aware ML practices (MLflow + containerization)
