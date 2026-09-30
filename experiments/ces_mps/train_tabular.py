"""Matched-input tabular controls, deltas, transitions and cross-fit residuals.

Fits all eligible training respondents. Boosting continues until held-out tune
loss fails to improve for four successive 25-tree extensions; no score gate
controls whether a different experiment branch runs.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits
from build_dataset import HERE,DATA,TARGETS,NUM,CAT,SEED
from scoring import score_frame

OUT=HERE/'results/tabular'
MODELS=HERE/'checkpoints/tabular'


def save_json(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,default=float));tmp.replace(path)


def aligned_prob(model,x,target,last=None):
    labels=np.array([0,1] if target=='c7010' else [1,2,3,4,5])
    raw=model.predict_proba(x)
    prob=np.zeros((len(x),len(labels)),dtype=np.float64)
    if last is None:
        for j,c in enumerate(model.classes_):prob[:,np.where(labels==c)[0][0]]=raw[:,j]
    else:
        for j,delta in enumerate(model.classes_):
            destination=last+delta
            for k,c in enumerate(labels):prob[:,k]+=raw[:,j]*(destination==c)
        # Impossibility correction is structural: discard probability assigned
        # to transitions outside the known survey answer domain.
        denom=prob.sum(1,keepdims=True)
        for k,c in enumerate(labels):prob[:,k]=np.where(denom[:,0]>0,prob[:,k],last==c)
    prob/=np.maximum(prob.sum(1,keepdims=True),1e-15)
    return prob


def predict(model,x,target,mode,last,base=None):
    if target in NUM:
        y=model.predict(x)
        if mode=='delta':y=y+last
        elif mode=='residual':y=y+base.predict(x)
        return np.clip(y,-100,100),None
    prob=aligned_prob(model,x,target,last if mode=='transition' else None)
    labels=np.array([0,1] if target=='c7010' else [1,2,3,4,5])
    return labels[prob.argmax(1)],prob


def converged_boost(x,y,xv,yv,target,mode,last,lastv,path,base=None,oof=None):
    if target in NUM:
        model=HistGradientBoostingRegressor(loss='absolute_error',max_iter=25,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,learning_rate=.05,early_stopping=False,warm_start=True,random_state=SEED)
        fit_y=y-last if mode=='delta' else y-oof if mode=='residual' else y
    else:
        model=HistGradientBoostingClassifier(max_iter=25,max_leaf_nodes=15,min_samples_leaf=40,l2_regularization=1.,learning_rate=.05,early_stopping=False,warm_start=True,random_state=SEED)
        fit_y=y-last if mode=='transition' else y
    best=float('inf');stale=0;history=[];start=time.monotonic()
    while True:
        model.fit(x,fit_y)
        p,probs=predict(model,xv,target,mode,lastv,base)
        score=float(np.mean(abs(p-yv))) if target in NUM else float(log_loss(yv,probs,labels=[0,1] if target=='c7010' else [1,2,3,4,5]))
        improved=score<best
        if improved:
            best=score;stale=0
            joblib.dump({'model':model,'base':base,'mode':mode,'target':target},path)
        else:stale+=1
        record=dict(iteration=model.n_iter_,tune_loss=score,best_tune_loss=best,improved=improved,seconds=time.monotonic()-start)
        history.append(record)
        save_json(path.with_suffix('.history.json'),history)
        print('FIT',path.stem,json.dumps(record),flush=True)
        if stale>=4:break
        model.set_params(max_iter=model.max_iter+25)
    return joblib.load(path)


def crossfit_ridge(x,y,ids,path):
    group=np.array([int(hashlib.sha256(str(i).encode()).hexdigest()[8:16],16)%5 for i in ids])
    oof=np.empty(len(y),dtype=np.float32)
    for fold in range(5):
        fit=group!=fold;val=~fit
        model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
        model.fit(x[fit],y[fit]);oof[val]=model.predict(x[val])
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),Ridge(alpha=100.))
    model.fit(x,y)
    joblib.dump({'model':model,'oof':oof,'fold':group},path)
    return model,oof


def emit(d,target,method,model,columns=None,mode='level',base=None):
    paths=[]
    trained_ids=set(d.loc[d.split.eq('train')&d.train_eligible_person&d[target].notna()&d[target+'_lag1'].notna(),'a0010']) if method not in ['last_observation','history3'] else set()
    for split in ['tune','select','retrospective_2025','retrospective_2026']:
        part=d[d.split.eq(split)].copy()
        # Primary paired estimand requires an observed previous answer. This
        # is an input-availability rule, and applies to ALL competing methods.
        part=part[part[target+'_lag1'].notna()]
        last=part[target+'_lag1'].to_numpy()
        if method=='last_observation':
            p=last;prob=None
            if target in CAT:
                labels=[0,1] if target=='c7010' else [1,2,3,4,5]
                prob=np.column_stack([p==c for c in labels]).astype(float)
        elif method=='history3':
            hist=part[[target+f'_lag{k}' for k in [1,2,3]]].to_numpy()
            if target in NUM:p=np.nanmedian(hist,axis=1);prob=None
            else:
                labels=np.array([0,1] if target=='c7010' else [1,2,3,4,5])
                prob=np.column_stack([np.mean(hist==c,axis=1) for c in labels]);prob/=prob.sum(1,keepdims=True)
                p=labels[prob.argmax(1)]
                # Ties prefer the most recent answer, as predeclared.
                ix=np.searchsorted(labels,last)
                tied=prob[np.arange(len(prob)),ix]==prob.max(1);p[tied]=last[tied]
        elif method=='ridge':p=np.clip(model.predict(part[columns].to_numpy()),-100,100);prob=None
        else:p,prob=predict(model,part[columns].to_numpy(),target,mode,last,base)
        out=part[['case_id','a0010','a0020','a0030','split','unseen_person','wgt']].copy()
        out['truth']=part[target];out['previous']=last;out['prediction']=p
        out['seen_in_train']=part.a0010.isin(trained_ids)
        if prob is not None:
            for j,c in enumerate([0,1] if target=='c7010' else [1,2,3,4,5]):out[f'p_{c}']=prob[:,j]
        path=OUT/f'{method}__{target}__{split}.parquet';out.to_parquet(path,index=False);paths.append(path.name)
        scores=score_frame(out,target)
        for row in scores:row.update(method=method,target=target,split=split)
        save_json(path.with_suffix('.metrics.json'),scores)
    return paths


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=TARGETS);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    schema=json.loads((DATA/'feature_schema.json').read_text())
    d=pd.read_parquet(DATA/'panel.parquet')
    d=d[d.eligible].copy()
    hashes=json.loads((DATA/'dataset_hashes.json').read_text())
    save_json(OUT/'run_spec.json',dict(dataset_hashes=hashes,seed=SEED,device='cpu',fit_weighting='unweighted individual loss',primary_population='all 3-consecutive-history cases with target-specific previous answer observed; current missing truth excluded by scorer only',convergence='25-tree extensions; stop after four extensions with no new best tune loss; best checkpoint saved',tune_loss='numeric MAE; categorical proper log loss on legal destination classes',residual='respondent-group 5-fold out-of-fold Ridge predictions on identical train rows; final Ridge fit on all train rows',methods=['last_observation','history3','ridge','hgb_basic_level','hgb_rich_level','hgb_rich_macro_level','hgb_rich_macro_delta','hgb_rich_macro_transition','hgb_rich_macro_residual']))
    for target in [args.target] if args.target else TARGETS:
        for method in ['last_observation','history3']:
            done=OUT/f'{method}__{target}.done.json'
            if not done.exists():save_json(done,dict(paths=emit(d,target,method,None)))
        valid=d[target].notna()&d[target+'_lag1'].notna()
        tr=d[valid&d.split.eq('train')&d.train_eligible_person]
        va=d[valid&d.split.eq('tune')]
        methods=[(group,'level') for group in schema]+[('rich_macro','delta' if target in NUM else 'transition')]
        if target in NUM:methods += [('rich_macro','residual')]
        for group,mode in methods:
            method=f'hgb_{group}_{mode}';path=MODELS/f'{method}__{target}.joblib';done=OUT/f'{method}__{target}.done.json'
            if done.exists():continue
            cols=schema[group]
            x=tr[cols].to_numpy();xv=va[cols].to_numpy();y=tr[target].to_numpy();yv=va[target].to_numpy()
            last=tr[target+'_lag1'].to_numpy();lastv=va[target+'_lag1'].to_numpy()
            print('START',method,target,'train',len(tr),'tune',len(va),flush=True)
            base=None;oof=None
            if mode=='residual':
                bpath=MODELS/f'ridge_oof__{target}.joblib'
                if bpath.exists():b=joblib.load(bpath);base,oof=b['model'],b['oof']
                else:base,oof=crossfit_ridge(x,y,tr.a0010.to_numpy(),bpath)
                rdone=OUT/f'ridge__{target}.done.json'
                if not rdone.exists():save_json(rdone,dict(paths=emit(d,target,'ridge',base,cols)))
            # An interrupted fit is rerun deterministically unless its final
            # marker is present; a partial best checkpoint is never mislabeled done.
            trained=converged_boost(x,y,xv,yv,target,mode,last,lastv,path,base,oof)
            paths=emit(d,target,method,trained['model'],cols,mode,trained['base'])
            save_json(done,dict(paths=paths,checkpoint=str(path),train_n=len(tr),tune_n=len(va),features=cols))
            print('DONE',method,target,flush=True)
    print('TABULAR_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
