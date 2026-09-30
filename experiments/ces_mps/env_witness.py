"""Seeded real MPS forward/backward witness; no CPU operator fallback."""
import argparse
import hashlib
import importlib.metadata as im
import json
import os
from pathlib import Path
import sys
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen3_5ForCausalLM

HERE = Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='Qwen/Qwen3-0.6B')
    a=ap.parse_args()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
    assert torch.backends.mps.is_available()
    torch.manual_seed(20260930)
    torch.set_num_threads(4)
    x=torch.randn(32,32)
    error=((x@x.T)-(x.to('mps')@x.T.to('mps')).cpu()).abs().max().item()
    torch.testing.assert_close(x@x.T,(x.to('mps')@x.T.to('mps')).cpu(),atol=1e-4,rtol=1e-4)
    print('KERNEL_WITNESS mps seeded_matmul_pass',flush=True)
    manifest=json.loads((HERE/'model_manifest.json').read_text()) if (HERE/'model_manifest.json').exists() else {}
    cached_probe=Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
    path=manifest.get(a.model,{}).get('path',str(cached_probe) if a.model=='Qwen/Qwen3-0.6B' else a.model)
    start=time.monotonic()
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    cls=Qwen3_5ForCausalLM if 'Qwen3.5' in a.model else AutoModelForCausalLM
    model=cls.from_pretrained(path,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('mps')
    assert {p.device.type for p in model.parameters()}=={'mps'}
    text=tok.apply_chat_template([{'role':'user','content':'Reply with the single digit that equals 3 plus 4.'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    x=tok(text,return_tensors='pt').to('mps')
    model.eval()
    with torch.no_grad(): base=model(**x,use_cache=False,logits_to_keep=1).logits.detach().float()
    base_grad=model(**x,use_cache=False,logits_to_keep=1).logits.detach().float()
    torch.testing.assert_close(base,base_grad,atol=0.02,rtol=0.02)
    mode_gap=(base-base_grad).abs().max().item()
    model=get_peft_model(model,LoraConfig(r=4,lora_alpha=8,lora_dropout=0.0,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    y=torch.tensor([tok.encode('7',add_special_tokens=False)[0]],device='mps')
    model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    z=model(**x,use_cache=False,logits_to_keep=1).logits[:,-1].float()
    loss=torch.nn.functional.cross_entropy(z,y)
    assert torch.isfinite(loss)
    loss.backward()
    grad=sum(p.grad.float().square().sum().item() for p in model.parameters() if p.grad is not None)
    assert 0<grad<float('inf')
    optimizer.step();torch.mps.synchronize()
    result=dict(model=a.model,path=str(path),device='mps',dtype='bfloat16',fallback=False,
        versions={p:im.version(p) for p in ['torch','transformers','peft','accelerate','huggingface-hub']},
        python=sys.version,kernel_max_error=error,grad_no_grad_max_error=mode_gap,
        loss=float(loss.item()),gradient_square_sum=grad,seconds=time.monotonic()-start,
        mps_allocated_bytes=torch.mps.current_allocated_memory(),mps_driver_bytes=torch.mps.driver_allocated_memory())
    out=HERE/'results';out.mkdir(exist_ok=True)
    dest=out/('witness_'+a.model.split('/')[-1]+'.json')
    dest.write_text(json.dumps(result,indent=2))
    print('MODEL_WITNESS mps finite_loss_nonzero_gradient '+str(dest),flush=True)

if __name__=='__main__': main()
