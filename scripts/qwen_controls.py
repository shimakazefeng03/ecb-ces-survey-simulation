"""Same-1024-label tabular/probability controls and paired Qwen comparisons."""
import json
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import HistGradientBoostingClassifier
from prepare_data import ROOT, SEED
from run_baselines import features

def main():
    d=pd.read_pickle(ROOT/'data/processed/lagged.pkl')
    js=lambda name:json.loads((ROOT/f'data/processed/qwen_{name}.json').read_text())
    tr=js('train');va=js('valid');te=js('test')
    train=d.loc[[int(r['id'][5:]) for r in tr]]
    valid=d.loc[[int(r['id'][5:]) for r in va]]
    sample=pd.read_pickle(ROOT/'data/processed/test_sample.pkl').set_index('case_id').loc[[r['id'] for r in te]]
    assert (train.c3110.to_numpy()-1==[r['label'] for r in tr]).all()
    assert (valid.c3110.to_numpy()-1==[r['label'] for r in va]).all()
    model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=10,random_state=SEED)
    model.fit(features(train),train.c3110-1)
    vp=model.predict_proba(features(valid));tp=model.predict_proba(features(sample))
    assert list(model.classes_)==[0,1,2,3,4]
    prediction_sets={'hgb_1024':(vp,tp),'transition_1024':(np.array([r['soft_target'] for r in va]),np.array([r['soft_target'] for r in te]))}
    rows=[];allp={}
    for mode,(v,p) in prediction_sets.items():
        vy=valid.c3110.to_numpy()-1;y=sample.c3110.to_numpy()-1;vl=np.log(v.clip(1e-12));logits=np.log(p.clip(1e-12))
        fit=minimize_scalar(lambda x:-np.log(softmax(vl/np.exp(x),axis=1)[np.arange(len(vy)),vy].clip(1e-12)).mean(),bounds=(-4,4),method='bounded')
        for calibration,temp in [('raw',1.),('temperature',float(np.exp(fit.x)))]:
            pp=softmax(logits/temp,axis=1);pred=pp.argmax(axis=1);truth=np.eye(5)[y]
            rows.append({'model':mode,'calibration':calibration,'n':len(y),'temperature':temp,'accuracy':float((pred==y).mean()),'nll':float(-np.log(pp[np.arange(len(y)),y].clip(1e-12)).mean()),'brier':float(((pp-truth)**2).sum(axis=1).mean()),'jsd_probability_mean':float(jensenshannon(pp.mean(axis=0),truth.mean(axis=0),base=2)**2)})
        allp[mode]=p.argmax(axis=1)
    q=json.loads((ROOT/'results/qwen_predictions.json').read_text())
    for mode in ['base','hard','soft']:
        z={r['case_id']:r for r in q if r['mode']==mode and r['calibration']=='temperature'}
        allp[mode]=np.array([np.argmax(z[c]['probabilities']) for c in sample.index])
    paired=pd.read_pickle(ROOT/'results/paired_predictions.pkl')
    for mode in ['deepseek_history','history3','hist_gradient_boosting']:
        z=paired[(paired.target=='c3110')&(paired.method==mode)].set_index('case_id').loc[sample.index]
        allp[mode]=z.prediction.to_numpy()-1
    rng=np.random.default_rng(SEED);bootrows=[];slices=[]
    for mode,pred in allp.items():
        correct=(pred==y).astype(float)
        for scope,mask in [('pooled',np.ones(len(y),dtype=bool)),('unseen',sample.unseen_person.to_numpy()),('seen',~sample.unseen_person.to_numpy())]+[(str(w),sample.a0030.to_numpy()==w) for w in [73,75,78]]:
            slices.append({'model':mode,'scope':scope,'n':int(mask.sum()),'accuracy':correct[mask].mean()})
    for against in ['base','soft','hgb_1024','transition_1024','deepseek_history','history3','hist_gradient_boosting']:
        delta=(allp['hard']==y).astype(float)-(allp[against]==y).astype(float)
        num=np.zeros(2000);den=np.zeros(2000)
        for c in sample.a0020.unique():
            mask=sample.a0020.to_numpy()==c;unique,inv=np.unique(sample.a0010.to_numpy()[mask],return_inverse=True)
            sums=np.bincount(inv,weights=delta[mask]);counts=np.bincount(inv);choices=rng.integers(0,len(unique),(2000,len(unique)))
            num+=sums[choices].sum(axis=1);den+=counts[choices].sum(axis=1)
        bootrows.append({'a':'hard','b':against,'accuracy_gain':delta.mean(),'lo':np.quantile(num/den,.025),'hi':np.quantile(num/den,.975),'n':len(y)})
    pd.DataFrame(rows).to_csv(ROOT/'results/qwen_probability_controls.csv',index=False)
    pd.DataFrame(bootrows).to_csv(ROOT/'results/qwen_paired_bootstrap.csv',index=False)
    pd.DataFrame(slices).to_csv(ROOT/'results/qwen_slices.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(bootrows).to_string(index=False))

if __name__=='__main__':main()
