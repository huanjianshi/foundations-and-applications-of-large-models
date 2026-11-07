"""Explicit, opt-in first download for a new machine; not called by evaluation."""
import os,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dest',type=Path,default=ROOT/'models/minilm');a=p.parse_args()
 a.dest.mkdir(parents=True,exist_ok=True)
 os.environ['HF_HOME']=str(ROOT/'models/cache')
 from huggingface_hub import snapshot_download
 snapshot_download('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',revision='e8f8c211226b894fcb81acc59f3b34ba3efd5f42',local_dir=a.dest,allow_patterns=['*.json','model.safetensors','*.txt','1_Pooling/*','README.md'])
 print(a.dest.resolve())
