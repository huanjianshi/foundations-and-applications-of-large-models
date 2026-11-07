"""Explicit official model downloads authorized for this project's optimization."""
import os,json,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['HF_HOME']=str(ROOT/'models/cache')
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_OFFLINE']='0'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import HfApi,snapshot_download

if __name__=='__main__':
    records=[]
    for repo,slug in [('BAAI/bge-small-zh-v1.5','bge-small-zh-v1.5'),('BAAI/bge-reranker-base','bge-reranker-base')]:
        revision={'BAAI/bge-small-zh-v1.5':'7999e1d3359715c523056ef9478215996d62a620','BAAI/bge-reranker-base':'2cfc18c9415c912f9d8155881c133215df768a70'}[repo]
        info=HfApi().model_info(repo,revision=revision,files_metadata=True)
        names=[f.rfilename for f in info.siblings]
        weights='model.safetensors' if 'model.safetensors' in names else 'pytorch_model.bin'
        allowed=[weights,'*.json','*.txt','*.model','README.md','LICENSE','1_Pooling/*']
        print(f'Downloading {repo} revision={info.sha} weights={weights}',flush=True)
        target=ROOT/'models'/slug
        started=time.time()
        snapshot_download(repo,revision=info.sha,local_dir=target,allow_patterns=allowed,max_workers=3)
        weight_file=target/weights
        digest=hashlib.sha256()
        with weight_file.open('rb') as handle:
            for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
        sibling=next(f for f in info.siblings if f.rfilename==weights)
        expected=sibling.lfs.sha256 if sibling.lfs else None
        if expected and digest.hexdigest()!=expected:
            raise RuntimeError(f'Official LFS hash mismatch: {repo}')
        records.append(dict(repo=repo,revision=info.sha,weights=weights,weight_bytes=weight_file.stat().st_size,directory_bytes=sum(p.stat().st_size for p in target.rglob('*') if p.is_file()),weight_sha256=digest.hexdigest(),matches_official_lfs=digest.hexdigest()==expected if expected else None,seconds=round(time.time()-started,2),official_source='https://huggingface.co/'+repo))
        (ROOT/'models/model_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Ready',slug,flush=True)
