"""Simple persistence, learned tabular, and conditional donor comparisons."""
from prepare_data import ROOT,TARGETS,NUM,CAT,SEED
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def features(df):
    cols=['a1010_age_prec_lag1','b7040_quintile_lag1','a1020_prec']+[f'{t}_lag{j}' for j in [1,2,3] for t in TARGETS]
    x=df[cols].copy()
    for c in ['AT','BE','DE','EL','ES','FI','FR','IE','IT','NL','PT']:x['country_'+c]=(df.a0020==c).astype(float)
    return x

def main():
    df=pd.read_pickle(ROOT/'data/processed/lagged.pkl')
    train=df[df.eligible & df.a0030.le(72) & ~df.unseen_person].copy()
    sample=pd.read_pickle(ROOT/'data/processed/test_sample.pkl')
    frame=df[df.eligible & df.a0030.isin([73,75,78])].copy().reset_index(drop=True)
    frame['case_id']=['FRAME%06d'%i for i in range(len(frame))]
    frame=frame.merge(sample[['a0010','a0030','case_id']].rename(columns={'case_id':'sample_case'}),on=['a0010','a0030'],how='left')
    frame['case_id']=frame.sample_case.fillna(frame.case_id)
    x=features(train);xt=features(frame); records=[]; rng=np.random.default_rng(SEED)
    imp=SimpleImputer(add_indicator=True);sc=StandardScaler()
    xs=sc.fit_transform(imp.fit_transform(x));xst=sc.transform(imp.transform(xt))
    # Donors restricted to same country; their outcomes are from 2025 only.
    donor_ix=np.zeros(len(frame),dtype=int)
    for country in sorted(train.a0020.unique()):
        ii=np.where(train.a0020.to_numpy()==country)[0];jj=np.where(frame.a0020.to_numpy()==country)[0]
        nn=NearestNeighbors(n_neighbors=20).fit(xs[ii]);near=nn.kneighbors(xst[jj],return_distance=False)
        donor_ix[jj]=ii[near[np.arange(len(jj)),rng.integers(0,20,len(jj))]]
    for target in TARGETS:
        y=train[target]; valid=y.notna()
        model=(HistGradientBoostingRegressor(loss='absolute_error',max_iter=100,max_leaf_nodes=15,l2_regularization=10,random_state=SEED)
               if target in NUM else HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=10,random_state=SEED))
        model.fit(x[valid],y[valid]);p=model.predict(xt)
        prev={}
        for wave in [73,75,78]:
            old=df[df.a0030.eq(wave-1)]
            for country,g in old.groupby('a0020'):
                v=g[target].dropna();prev[(wave,country)]=v.median() if target in NUM else v.mode().iloc[0]
        lag=frame[[f'{target}_lag{j}' for j in [1,2,3]]]
        roll=lag.median(axis=1).to_numpy() if target in NUM else lag.mode(axis=1)[0].to_numpy()
        country=np.array([prev[(w,c)] for w,c in zip(frame.a0030,frame.a0020)])
        pred={'country_previous':country,'last_observation':frame[f'{target}_lag1'].to_numpy(),'history3':roll,'hist_gradient_boosting':p,
              'conditional_donor':train.iloc[donor_ix][target].to_numpy()}
        for method,v in pred.items():
            # A missing historical numeric response falls back to country prior; no target imputation.
            v=np.where(np.isfinite(v),v,country)
            z=frame[['case_id','a0010','a0030','a0020','unseen_person','wgt']].copy()
            z['target']=target;z['method']=method;z['truth']=frame[target].to_numpy();z['prediction']=v
            records.append(z)
        print(target,'trained',int(valid.sum()),'predicted',len(frame),flush=True)
    pd.concat(records).to_pickle(ROOT/'results/baseline_full.pkl')
    meta={'train_rows':len(train),'train_people':train.a0010.nunique(),'train_waves':sorted(train.a0030.unique().tolist()),'test_frame_rows':len(frame),'hyperparameters':{'max_iter':100,'max_leaf_nodes':15,'l2_regularization':10},'donor_k':20}
    (ROOT/'results/baseline_design.json').write_text(json.dumps(meta,indent=2))
    frame.to_pickle(ROOT/'data/processed/test_frame.pkl')

if __name__=='__main__':main()
