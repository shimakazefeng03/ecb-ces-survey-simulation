"""Local MLX first-token classification and two small LoRA objectives."""
from pathlib import Path
import json,time,argparse,sys,hashlib
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.models.base import create_attention_mask,create_ssm_mask

ROOT=Path(__file__).resolve().parents[1]
MODEL='/Users/fengyang/Desktop/research/new/experiments/qwen35_mlx/checkpoint'

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['base','hard','soft'],required=True);p.add_argument('--seed',type=int,default=20260930);a=p.parse_args()
    mx.set_default_device(mx.gpu);mx.set_cache_limit(256*1024**2);mx.random.seed(a.seed)
    model,tok=load(MODEL);model.eval()
    ids=[tok.encode(str(i),add_special_tokens=False) for i in range(1,6)]
    assert all(len(i)==1 for i in ids)
    candidate=mx.array([i[0] for i in ids])
    # Tied embedding rows give exactly the five answer-token logits, without allocating a full-vocabulary output at every prompt token.
    embedding=model.language_model.model.embed_tokens
    answer_vectors=embedding(candidate);mx.eval(answer_vectors)
    def logits(m,x):
        if x.ndim==3:
            h=m.layers[-1](x,mask=create_attention_mask(x,None),cache=None)
            hidden=m.language_model.model.norm(h)[:,-1,:]
        else:hidden=m.language_model.model(x)[:,-1,:]
        return (hidden @ answer_vectors.T).astype(mx.float32)
    datasets={}
    for name in ['train','valid','test']:
        data=json.loads((ROOT/f'data/processed/qwen_{name}.json').read_text())
        for r in data:
            prompt=tok.apply_chat_template([{'role':'user','content':r['prompt']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            r['tokens']=tok.encode(prompt,add_special_tokens=False)
        datasets[name]=data
    outdir=ROOT/'results/qwen'/(a.mode if a.seed==20260930 else a.mode+'_seed'+str(a.seed));outdir.mkdir(parents=True,exist_ok=True)
    meta={'mode':a.mode,'seed':a.seed,'model_path':MODEL,'device':str(mx.default_device()),'device_info':mx.device_info(),'candidate_token_ids':ids,
          'train_n':len(datasets['train']),'prompt_tokens_range':[min(len(r['tokens']) for r in datasets['train']),max(len(r['tokens']) for r in datasets['train'])]}
    (outdir/'config.json').write_text(json.dumps(meta,indent=2,default=str))
    if a.mode!='base':
        # Only the final block is trainable. Cache the identical frozen prefix once
        # for both objectives, avoiding repeated evaluation of 31 unchanged blocks.
        cache_dir=ROOT/'data/processed/qwen_prefix';cache_dir.mkdir(exist_ok=True)
        count=0;checked=False
        for data in datasets.values():
            for r in data:
                digest=hashlib.sha256(json.dumps(r['tokens']).encode()).hexdigest()
                path=cache_dir/(digest+'.safetensors');r['prefix_path']=path
                if path.exists():continue
                tokens=mx.array([r['tokens']]);h=embedding(tokens)
                fa=create_attention_mask(h,None);ssm=create_ssm_mask(h,None)
                for layer in model.layers[:-1]:h=layer(h,mask=ssm if layer.is_linear else fa,cache=None)
                mx.eval(h)
                if not checked:
                    gap=float(mx.max(mx.abs(logits(model,tokens)-logits(model,h))).item())
                    assert gap<.05,gap
                    meta['cached_prefix_logit_max_abs_error']=gap;checked=True
                mx.save_safetensors(str(path),{'hidden':h});count+=1;mx.clear_cache()
                if count%50==0:print('cached frozen prefix',count,flush=True)
        meta['prefix_cache']='31 frozen blocks; reused exactly across hard and soft objectives'
        (outdir/'config.json').write_text(json.dumps(meta,indent=2,default=str))
    def inputs(r):
        return mx.load(str(r['prefix_path']))['hidden'] if 'prefix_path' in r else mx.array([r['tokens']])
    def predict(name,step):
        path=outdir/f'{name}_{step}.jsonl';done=set()
        if path.exists():done={json.loads(x)['id'] for x in path.read_text().splitlines()}
        model.eval()
        with path.open('a') as f:
            for i,r in enumerate(datasets[name]):
                if r['id'] in done:continue
                z=logits(model,inputs(r))[0];pr=mx.softmax(z);mx.eval(pr)
                f.write(json.dumps({'id':r['id'],'label':r['label'],'logits':z.tolist(),'probabilities':pr.tolist()})+'\n');f.flush()
                if (i+1)%50==0:print(a.mode,name,step,i+1,flush=True)
                mx.clear_cache()
    if a.mode=='base':
        predict('valid',0);predict('test',0);return
    # Refresh targets after prefix caching; prompts and labels are unchanged.
    datasets['train']=[dict(r,soft_target=new['soft_target']) for r,new in zip(datasets['train'],json.loads((ROOT/'data/processed/qwen_train.json').read_text()))]
    model.freeze()
    mx.random.seed(a.seed)
    config={'rank':8,'scale':16.0,'dropout':0.0,'keys':['self_attn.q_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']}
    linear_to_lora_layers(model,1,config)
    meta['lora']=config;meta['num_layers']=1;meta['trainable_parameters']=sum(v.size for _,v in tree_flatten(model.trainable_parameters()))
    (outdir/'config.json').write_text(json.dumps(meta,indent=2,default=str))
    optimizer=optim.Adam(learning_rate=5e-5)
    def loss(m,x,y):
        logp=nn.log_softmax(logits(m,x),axis=-1)
        return -(logp*y).sum()
    value_grad=nn.value_and_grad(model,loss)
    rng=np.random.default_rng(a.seed);order=rng.permutation(len(datasets['train']))
    start=time.time();model.train();losses=[]
    with (outdir/'training.jsonl').open('w') as f:
        for step,j in enumerate(order,1):
            r=datasets['train'][j]
            y=r['soft_target'] if a.mode=='soft' else np.eye(5)[r['label']].tolist()
            l,grad=value_grad(model,inputs(r),mx.array([y]));optimizer.update(model,grad)
            mx.eval(model.parameters(),optimizer.state,l);losses.append(float(l.item()));mx.clear_cache()
            if step%32==0:
                row={'step':step,'mean_loss_last32':float(np.mean(losses[-32:])),'seconds':time.time()-start,'peak_gb':mx.get_peak_memory()/1e9}
                f.write(json.dumps(row)+'\n');f.flush();print(a.mode,row,flush=True)
            if step in [256,512,1024]:
                mx.save_safetensors(str(outdir/f'adapters_{step}.safetensors'),dict(tree_flatten(model.trainable_parameters())))
                predict('valid',step);model.train()
    # Checkpoint selection uses only 2025 validation log loss.
    scores={}
    for step in [256,512,1024]:
        z=[json.loads(x) for x in (outdir/f'valid_{step}.jsonl').read_text().splitlines()]
        scores[step]=float(np.mean([-np.log(max(r['probabilities'][r['label']],1e-12)) for r in z]))
    best=min(scores,key=scores.get);model.load_weights(str(outdir/f'adapters_{best}.safetensors'),strict=False)
    model.eval();gaps=[]
    for r in datasets['test'][:3]:
        gaps.append(float(mx.max(mx.abs(logits(model,mx.array([r['tokens']]))-logits(model,inputs(r)))).item()))
    assert max(gaps)<.05,gaps
    (outdir/'selection.json').write_text(json.dumps({'validation_nll':scores,'selected_step':best,'training_seconds':time.time()-start,'cached_vs_full_logit_max_errors':gaps},indent=2))
    predict('test',best)

if __name__=='__main__':main()
