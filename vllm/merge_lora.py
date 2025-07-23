import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
import warnings
import torch

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_path = os.path.join(base_dir, "adapter")       
    merged_save_path = os.path.join(base_dir, "merged_llama3")  

    # ✅ 載入 base 模型（不使用 device_map）
    base = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        torch_dtype=torch.float16,
        device_map=None,  # 直接丟給 GPU，如果爆了就換手動切分
        token=HF_TOKEN,
        trust_remote_code=True
    )

    print(f"[INFO] Loading LoRA adapter from {adapter_path} ...")

    # ✅ base_model_prefix 一定要 "model"（剛剛模型印出來就是）
    adapter = PeftModel.from_pretrained(
        base,
        adapter_path,
        is_trainable=False,
        base_model_prefix="model"
    )

    print("[INFO] Merging adapter into base model ...")
    merged = adapter.merge_and_unload()

    print(f"[INFO] Saving merged model to {merged_save_path} ...")
    merged.save_pretrained(merged_save_path)

    print(f"[INFO] Saving tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        token=HF_TOKEN
    )
    tokenizer.save_pretrained(merged_save_path)

    print(f"[INFO] ✅ Merged model and tokenizer saved at {merged_save_path}")

if __name__ == "__main__":
    main()
