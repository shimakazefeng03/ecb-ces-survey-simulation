"""Fixed scorers; predictions and truths meet only here."""
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import f1_score, log_loss
from build_dataset import NUM


def metrics(y,p,w,target,prob=None):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);w=np.asarray(w,dtype=float)
    eligible=np.isfinite(y)&np.isfinite(w)&(w>0)
    observed=eligible&np.isfinite(p)
    out={'n_truth':int(eligible.sum()),'n_scored':int(observed.sum()),'failure_rate':float(1-observed.sum()/max(eligible.sum(),1))}
    y,p,w=y[observed],p[observed],w[observed]
    if not len(y):return out
    w=w/w.sum()
    if target in NUM:
        out.update(mae=float(w@abs(y-p)),rmse=float(np.sqrt(w@((y-p)**2))),bias=float(w@(p-y)),wasserstein=float(wasserstein_distance(y,p,u_weights=w,v_weights=w)),truth_mean=float(w@y),prediction_mean=float(w@p),truth_std=float(np.sqrt(w@((y-w@y)**2))),prediction_std=float(np.sqrt(w@((p-w@p)**2))))
        def q(x):
            ix=np.argsort(x);return np.interp([.25,.75],np.cumsum(w[ix])-.5*w[ix],x[ix])
        yi=np.diff(q(y))[0];pi=np.diff(q(p))[0]
        out['iqr_ratio']=float(pi/yi) if yi>0 else None
    else:
        classes=np.array([0,1] if target=='c7010' else [1,2,3,4,5])
        out.update(accuracy=float(w@(y==p)),macro_f1=float(f1_score(y,p,labels=classes,average='macro',sample_weight=w,zero_division=0)),ordinal_mae=float(w@abs(y-p)))
        yp=np.array([w@(y==c) for c in classes]);pp=np.array([w@(p==c) for c in classes])
        out['jsd']=float(jensenshannon(yp,pp,base=2)**2)
        if prob is not None:
            prob=np.asarray(prob,dtype=float)[observed]
            assert np.isfinite(prob).all() and (prob>=0).all()
            prob=prob/np.maximum(prob.sum(1,keepdims=True),1e-15)
            idx=np.searchsorted(classes,y)
            onehot=np.eye(len(classes))[idx]
            out['brier']=float(w@np.sum((prob-onehot)**2,axis=1))
            out['nll']=float(w@(-np.log(prob[np.arange(len(y)),idx].clip(1e-12))))
            confidence=prob.max(1);correct=classes[prob.argmax(1)]==y
            ece=0.
            for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
                m=(confidence>=lo)&(confidence<hi if hi<1 else confidence<=hi)
                if m.any():ece+=abs(np.sum(w[m]*(correct[m]-confidence[m])))
            out['ece10']=float(ece)
    return out


def score_frame(frame,target):
    probcols=[c for c in frame if c.startswith('p_')]
    rows=[]
    strata={'all':np.ones(len(frame),dtype=bool),'person_split_train_allowed':~frame.unseen_person.to_numpy(),'person_split_train_excluded':frame.unseen_person.to_numpy()}
    if 'seen_in_train' in frame:
        strata['seen_in_supervised_train']=frame.seen_in_train.to_numpy()
        strata['absent_from_supervised_train']=~frame.seen_in_train.to_numpy()
    observed=frame.truth.notna()&frame.previous.notna()
    strata['changed']=np.asarray(observed&frame.truth.ne(frame.previous));strata['unchanged']=np.asarray(observed&frame.truth.eq(frame.previous))
    for k in ['a0020','a0030']:
        for val in frame[k].unique():strata[f'{k}={val}']=frame[k].eq(val).to_numpy()
    for stratum,mask in strata.items():
        part=frame.loc[mask]
        if not len(part):continue
        for weighted in [False,True]:
            out=metrics(part.truth,part.prediction,part.wgt if weighted else np.ones(len(part)),target,part[probcols].to_numpy() if probcols else None)
            rows.append(dict(stratum=stratum,weighted=weighted,**out))
    return rows
