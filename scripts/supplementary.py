"""Missingness, weighting, repeatability and structural diagnostics."""
import json
import numpy as np
import pandas as pd
from prepare_data import ROOT,TARGETS,NUM,SEED
from analyze import metrics

def main():
    x=pd.read_pickle(ROOT/'results/paired_predictions.pkl');s=pd.read_pickle(ROOT/'data/processed/test_sample.pkl');out=[]
    for t in TARGETS:
        a=x[(x.target==t)&(x.method=='deepseek_history')]
        b=x[(x.target==t)&(x.method=='deepseek_history_repeat')]
        j=a.merge(b,on='case_id',suffixes=('_a','_b')).dropna(subset=['prediction_a','prediction_b'])
        out.append({'target':t,'n':len(j),'exact_repeat_rate':float((j.prediction_a==j.prediction_b).mean()),'mean_abs_repeat_difference':float(abs(j.prediction_a-j.prediction_b).mean())})
    pd.DataFrame(out).to_csv(ROOT/'results/repeatability.csv',index=False)
    rows=[]
    for (target,method),g in x.groupby(['target','method']):
        g=g.copy();g['analysis_weight']=g.analysis_weight.clip(upper=g.analysis_weight.quantile(.95))
        rows.append({'target':target,'method':method,'scope':'pooled_95pct_weight_cap',**metrics(g,True)})
    pd.DataFrame(rows).to_csv(ROOT/'results/weight_sensitivity.csv',index=False)
    # Compare every method on the same non-abstention set for spending.
    z=x[x.target=='c6120'];main=['country_previous','last_observation','history3','hist_gradient_boosting','conditional_donor','deepseek_demo','deepseek_history','deepseek_shuffled_history']
    pivot=z[z.method.isin(main)].pivot(index='case_id',columns='method',values='prediction');common=pivot.dropna().index
    rows=[{'target':t,'method':m,**metrics(g)} for (t,m),g in z[z.case_id.isin(common)&z.method.isin(main)].groupby(['target','method'])]
    pd.DataFrame(rows).to_csv(ROOT/'results/spending_common_cases.csv',index=False)
    # Compare marginal distributions with within-person cross-target structure.
    structure=[]
    for method,g in x.groupby('method'):
        p=g.pivot(index='case_id',columns='target',values='prediction');y=g.pivot(index='case_id',columns='target',values='truth')
        structure.append({'method':method,'synthetic_inflation_horizon_spearman':p[['c1120','c1220']].corr(method='spearman').iloc[0,1],
                          'real_inflation_horizon_spearman':y[['c1120','c1220']].corr(method='spearman').iloc[0,1],
                          'synthetic_horizon_identical':float((p.c1120==p.c1220).mean()),'real_horizon_identical':float((y.c1120==y.c1220).mean())})
    pd.DataFrame(structure).to_csv(ROOT/'results/structure.csv',index=False)
    # Target-vs-lag distinction and split audit are executable invariants.
    d=pd.read_pickle(ROOT/'data/processed/lagged.pkl');eligible=d[d.eligible]
    weight_rows=[]
    for wave,g in s.groupby('a0030'):
        w=g.analysis_weight.to_numpy();n=int((d.a0030==wave).sum());ne=int((eligible.a0030==wave).sum())
        weight_rows.append({'wave':wave,'sample':len(g),'eligible':ne,'full_wave':n,'retention_share':ne/n,'kish_ess':w.sum()**2/(w*w).sum(),'weight_cv':w.std()/w.mean(),'max_weight_share':w.max()/w.sum()})
    pd.DataFrame(weight_rows).to_csv(ROOT/'results/weight_audit.csv',index=False)
    assert all((eligible[f'a0030_lag{j}']==eligible.a0030-j).all() for j in [1,2,3])
    train=eligible[(eligible.a0030<=72)&~eligible.unseen_person]
    assert set(train.a0010).isdisjoint(set(s[s.unseen_person].a0010))
    assert len(s)==660 and not s[['a0010','a0030']].duplicated().any()
    known=set(zip(s.case_id,['']*len(s)))
    raw=[json.loads(l) for l in (ROOT/'results/deepseek_raw.jsonl').read_text().splitlines()]
    valid={(r['case_id'],r['arm']):r for r in raw if 'answers' in r}
    for arm in ['demo','history','shuffled_history','history1','matched_history','reversed_history']:
        assert sum(a==arm for _,a in valid)==660,(arm,'incomplete')
    (ROOT/'results/integrity_checks.json').write_text(json.dumps({'lag_timing':'passed','unseen_training_ids_disjoint':'passed','paired_sample_keys':'passed','six_prompt_arms_complete':'passed','forbidden_wording_archive_excluded':'passed_by_explicit_input_path','note':'Retrospective release and pretrained contamination remain limitations, not testable invariants.'},indent=2))
    print('Supplementary metrics and six-arm completeness checks passed')

if __name__=='__main__':main()
