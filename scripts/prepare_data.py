"""Build lag-only CES features and a reproducible paired test sample."""
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TARGETS=['c1120','c1220','c6120','c3010','c3110','c7010']
NUM=TARGETS[:3]; CAT=TARGETS[3:]
SEED=20260930

def main():
    cols=['a0010','a0020','a0030','a1010_age_prec','b7040_quintile','wgt','c1110','c1120','c1220','c3010','c3110','c3220','c4020','c6120','c7010']
    df=pd.concat([pd.read_csv(ROOT/f'data/raw/ecb.CES_data_{year}_monthly.en.csv',usecols=cols)[cols] for year in [2025,2026]],ignore_index=True)
    assert not df[['a0010','a0030']].duplicated().any()
    (ROOT/'data/processed').mkdir(parents=True,exist_ok=True)
    (ROOT/'results').mkdir(exist_ok=True)
    df.to_pickle(ROOT/'data/processed/panel.pkl')
    bg=pd.read_csv(ROOT/'data/raw/ecb.CES_data_background.en.csv',usecols=['a0010','a1020_prec','b2100_prec'])
    assert not bg.a0010.duplicated().any()
    df=df.merge(bg,on='a0010',how='left',validate='many_to_one')
    df.c6120=df.c6120.replace(-999,np.nan)
    df=df.sort_values(['a0010','a0030']).reset_index(drop=True)
    g=df.groupby('a0010',sort=False)
    features=['a1010_age_prec','b7040_quintile']
    for lag in [1,2,3]:
        for col in TARGETS+features+['a0030']:
            df[f'{col}_lag{lag}']=g[col].shift(lag)
    eligible=np.logical_and.reduce([df[f'a0030_lag{j}'].eq(df.a0030-j) for j in [1,2,3]])
    df['eligible']=eligible
    # Deterministic person split, independent of labels; same split across all waves.
    ids=df.a0010.unique()
    hold={v:int(hashlib.sha256(str(v).encode()).hexdigest()[:8],16)%5==0 for v in ids}
    df['unseen_person']=df.a0010.map(hold)
    df.to_pickle(ROOT/'data/processed/lagged.pkl')
    sample=[]
    for wave in [73,75,78]:
        frame=df[df.eligible & df.a0030.eq(wave)]
        for country,grp in frame.groupby('a0020'):
            s=grp.sample(n=min(20,len(grp)),random_state=SEED+wave)
            s=s.copy();s['sample_probability']=len(s)/len(grp)
            s['analysis_weight']=s.wgt/s.sample_probability
            sample.append(s)
    sample=pd.concat(sample).sort_values(['a0030','a0020','a0010']).reset_index(drop=True)
    sample['case_id']=['CES%04d'%i for i in range(len(sample))]
    sample.to_pickle(ROOT/'data/processed/test_sample.pkl')
    sample.drop(columns=['a0010']).to_csv(ROOT/'data/processed/test_sample_no_id.csv',index=False)
    out={'seed':SEED,'rows':len(df),'persons':df.a0010.nunique(),'eligible_by_wave':df[df.eligible].groupby('a0030').size().to_dict(),
         'test_n':len(sample),'test_unique_people':sample.a0010.nunique(),'test_unseen_n':int(sample.unseen_person.sum()),
         'test_counts':sample.groupby(['a0030','a0020']).size().to_string(),
         'missing_target':sample[TARGETS].isna().sum().to_dict(),'background_missing':df[['a1020_prec','b2100_prec']].isna().sum().to_dict()}
    (ROOT/'results/sample_audit.json').write_text(json.dumps(out,indent=2,default=int))
    print(json.dumps(out,default=int))

if __name__=='__main__':main()
