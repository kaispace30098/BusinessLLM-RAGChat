from huggingface_hub import HfApi
import os
from dotenv import load_dotenv

load_dotenv()


api = HfApi()
api.upload_folder(
    folder_path="./merged_llama3",  
    repo_id="tomc30098/llama3-8b-qlora-merged",  
    token=os.getenv('HF_TOKEN_'),  ## for huggingface hub
)
