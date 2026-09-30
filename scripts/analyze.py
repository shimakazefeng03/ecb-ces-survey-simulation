"""Matched evaluation, uncertainty, and finite-panel residual correction."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance,spearmanr,t as student_t
from scipy.spatial.distance import jensenshannon
from prepare_data import ROOT,TARGETS,NUM,CAT,SEED

def wquant(x,w,q):
    order=np.argsort(x);x=x[order];w=w[order]
    return np.interp(q,(np.cumsum(w)-.5*w)/w.sum(),x)

def metrics(g,weighted=False):
    valid=g.truth.notna() & g.prediction.notna();v=g[valid]
    y=v.truth.to_numpy();p=v.prediction.to_numpy();w=v.analysis_weight.to_numpy() if weighted else np.ones(len(v))
    r={'n':len(v),'n_truth':int(g.truth.notna().sum()),'response_coverage':float(valid.sum()/max(1,g.truth.notna().sum()))}
    if not len(v):return r
    r['bias']=float(np.average(p-y,weights=w))
    if g.target.iloc[0] in NUM:
        yq=wquant(y,w,[.25,.5,.75]);pq=wquant(p,w,[.25,.5,.75])
        r.update(mae=float(np.average(abs(p-y),weights=w)),rmse=float(np.sqrt(np.average((p-y)**2,weights=w))),
                 wasserstein=float(wasserstein_distance(y,p,u_weights=w,v_weights=w)),iqr_ratio=float((pq[2]-pq[0])/(yq[2]-yq[0])) if yq[2]>yq[0] else None,
                 real_median=float(yq[1]),synthetic_median=float(pq[1]),spearman=float(spearmanr(y,p).statistic),
                 real_zero=float(np.average(y==0,weights=w)),synthetic_zero=float(np.average(p==0,weights=w)),
                 real_integer=float(np.average(y==np.round(y),weights=w)),synthetic_integer=float(np.average(p==np.round(p),weights=w)))
    else:
        classes=[0,1] if g.target.iloc[0]=='c7010' else [1,2,3,4,5]
        a=np.array([w[y==c].sum() for c in classes]);b=np.array([w[p==c].sum() for c in classes])
        r.update(accuracy=float(np.average(p==y,weights=w)),jsd=float(jensenshannon(a,b,base=2)**2),
                 balanced_accuracy=float(np.mean([np.average(p[y==c]==c,weights=w[y==c]) for c in classes if np.any(y==c)])))
        if g.target.iloc[0] in ['c3010','c3110']:r['ordinal_mae']=float(np.average(abs(p-y),weights=w))
    return r

def load_predictions():
    s=pd.read_pickle(ROOT/'data/processed/test_sample.pkl')
    base=pd.read_pickle(ROOT/'results/baseline_full.pkl');b=base[base.case_id.isin(s.case_id)].copy()
    b=b.merge(s[['case_id','analysis_weight']],on='case_id',validate='many_to_one')
    records=[];raw={};errs=[]
    if (ROOT/'results/deepseek_raw.jsonl').exists():
        for line in (ROOT/'results/deepseek_raw.jsonl').read_text().splitlines():
            z=json.loads(line)
            if 'answers' in z:raw[(z['case_id'],z['arm'])]=z
            else:errs.append(z)
    for (cid,arm),z in raw.items():
        r=s[s.case_id.eq(cid)].iloc[0]
        for target in TARGETS:
            records.append({'case_id':cid,'a0010':r.a0010,'a0030':r.a0030,'a0020':r.a0020,'unseen_person':r.unseen_person,
                'wgt':r.wgt,'analysis_weight':r.analysis_weight,'target':target,'method':'deepseek_'+arm,'truth':r[target],'prediction':z['answers'][target]})
    full=pd.concat([b,pd.DataFrame(records)],ignore_index=True)
    full.to_pickle(ROOT/'results/paired_predictions.pkl')
    usage={}
    for z in raw.values():
        for k,v in z['response'].get('usage',{}).items():
            if isinstance(v,int):usage[k]=usage.get(k,0)+v
    (ROOT/'results/api_audit.json').write_text(json.dumps({'successful_by_arm':pd.Series([arm for _,arm in raw]).value_counts().to_dict(),'recorded_errors':len(errs),'valid_usage':usage,'returned_model_fields':sorted(set(z['response'].get('model','unknown') for z in raw.values())),'first_request_started_at':min(z['started_at'] for z in raw.values()),'last_request_started_at':max(z['started_at'] for z in raw.values())},indent=2))
    return full,base,s

def evaluate(full,base):
    rows=[]
    for (target,method),g in full.groupby(['target','method']):
        for weighting in [False,True]:
            rows.append({'target':target,'method':method,'scope':'pooled','weighted':weighting,**metrics(g,weighting)})
            for wave,w in g.groupby('a0030'):
                rows.append({'target':target,'method':method,'scope':str(wave),'weighted':weighting,**metrics(w,weighting)})
            for unseen,w in g.groupby('unseen_person'):
                rows.append({'target':target,'method':method,'scope':'unseen' if unseen else 'seen','weighted':weighting,**metrics(w,weighting)})
    pd.DataFrame(rows).to_csv(ROOT/'results/metrics.csv',index=False)
    r=[]
    for (target,method,wave),g in base.groupby(['target','method','a0030']):
        g=g.copy();g['analysis_weight']=g.wgt
        for weighted in [False,True]:r.append({'target':target,'method':method,'wave':wave,'weighted':weighted,**metrics(g,weighted)})
    pd.DataFrame(r).to_csv(ROOT/'results/full_frame_metrics.csv',index=False)
    # Country and public covariate slices: descriptive; small per-country n, not strong fairness inference.
    subgroup=[]
    for (target,method,country),g in full.groupby(['target','method','a0020']):subgroup.append({'target':target,'method':method,'country':country,**metrics(g)})
    pd.DataFrame(subgroup).to_csv(ROOT/'results/country_metrics.csv',index=False)

def paired_bootstrap(full):
    rng=np.random.default_rng(SEED);out=[]
    contrasts=[('deepseek_history',b) for b in ['deepseek_demo','deepseek_shuffled_history','last_observation','hist_gradient_boosting','history3','deepseek_history1','deepseek_matched_history','deepseek_reversed_history']]
    for target in TARGETS:
        t=full[full.target.eq(target)]
        for a,b in contrasts:
            x=t[t.method.eq(a)].merge(t[t.method.eq(b)],on='case_id',suffixes=('_a','_b'))
            x=x.dropna(subset=['truth_a','prediction_a','prediction_b'])
            if not len(x):continue
            y=x.truth_a.to_numpy();pa=x.prediction_a.to_numpy();pb=x.prediction_b.to_numpy()
            # Positive means A is better for both families.
            delta=abs(pb-y)-abs(pa-y) if target in NUM else (pa==y).astype(float)-(pb==y).astype(float)
            ids=x.a0010_a.to_numpy();countries=x.a0020_a.to_numpy();num=np.zeros(2000);den=np.zeros(2000)
            for c in np.unique(countries):
                mask=countries==c;unique,inv=np.unique(ids[mask],return_inverse=True)
                sums=np.bincount(inv,weights=delta[mask]);counts=np.bincount(inv)
                choices=rng.integers(0,len(unique),(2000,len(unique)))
                num+=sums[choices].sum(axis=1);den+=counts[choices].sum(axis=1)
            boot=num/den
            out.append({'target':target,'a':a,'b':b,'n':len(x),'positive_means':'A_better','gain':delta.mean(),'lo':np.quantile(boot,.025),'hi':np.quantile(boot,.975),'bootstrap':2000,'cluster':'person_within_country'})
    pd.DataFrame(out).to_csv(ROOT/'results/paired_bootstrap.csv',index=False)

def correction(full):
    """Fixed finite target panel. Labels are sampled WITHOUT replacement; no population inference."""
    rng=np.random.default_rng(SEED);rows=[]
    for target in ['c1120','c1220','c3110','c7010']:
        for wave in [73,75,78]:
            t=full[full.target.eq(target)&full.a0030.eq(wave)]
            truth=t.drop_duplicates('case_id').set_index('case_id').truth
            pred=t.pivot(index='case_id',columns='method',values='prediction').reindex(truth.index)
            methods=[c for c in ['last_observation','history3','hist_gradient_boosting','deepseek_demo','deepseek_history','deepseek_shuffled_history'] if c in pred and pred[c].notna().all()]
            y=truth.to_numpy();p=pred[methods].to_numpy()
            if target=='c3110':y=(y<=2).astype(float);p=(p<=2).astype(float) # economically interpretable adverse-outlook share
            N=len(y);mu=y.mean()
            for n in [55,110]:
                if N<=n:continue
                estimates={m:[] for m in ['human_only']+methods};covers={m:[] for m in estimates};widths={m:[] for m in estimates}
                for trial in range(1000):
                    ix=rng.choice(N,n,replace=False)
                    for j,method in enumerate(['human_only']+methods):
                        f=np.zeros(N) if method=='human_only' else p[:,j-1]
                        residual=y[ix]-f[ix]
                        est=f.mean()+residual.mean()
                        se=np.sqrt((1-n/N)*residual.var(ddof=1)/n)
                        half=student_t.ppf(.975,n-1)*se
                        estimates[method].append(est);covers[method].append(abs(est-mu)<=half);widths[method].append(2*half)
                human_var=np.var(estimates['human_only'],ddof=1)
                for method,e in estimates.items():
                    e=np.array(e);rows.append({'target':target,'wave':wave,'N':N,'label_n':n,'trials':1000,'method':method,'truth':mu,
                    'bias':e.mean()-mu,'rmse':np.sqrt(np.mean((e-mu)**2)),'coverage95':np.mean(covers[method]),'mean_ci_width':np.mean(widths[method]),
                    'variance_gain':human_var/np.var(e,ddof=1),'uncorrected_bias':(0 if method=='human_only' else p[:,methods.index(method)].mean()-mu)})
    pd.DataFrame(rows).to_csv(ROOT/'results/correction.csv',index=False)

if __name__=='__main__':
    full,base,s=load_predictions();evaluate(full,base);paired_bootstrap(full);correction(full)
    m=pd.read_csv(ROOT/'results/metrics.csv');print(m[(m.scope=='pooled') & ~m.weighted][['target','method','n','mae','accuracy','wasserstein','iqr_ratio']].to_string(index=False))
