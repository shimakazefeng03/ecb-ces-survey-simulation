"""Frozen, lag-only panel and chronological split. No label-dependent sampling."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
DATA=HERE/'data'
TARGETS=['c1120','c1220','c6120','c3010','c3110','c7010']
NUM=TARGETS[:3]
CAT=TARGETS[3:]
EXTRA=['c1020','c3220','c4020','c4030','c4031','emp_status']
DEMO=['a1010_age_prec','b7040_quintile']
SEED=20260930
SPLITS={'train':[4,48],'tune':[49,54],'select':[55,60],
        'retrospective_2025':[61,72],'retrospective_2026':[73,78]}

def main():
    cols=['a0010','a0020','a0030','wgt']+TARGETS+EXTRA+DEMO
    pieces=[]
    for year in range(2020,2027):
        p=DATA/'raw'/f'ecb.CES_data_{year}_monthly.en.csv'
        d=pd.read_csv(p,usecols=cols)
        pieces.append(d[cols])
    d=pd.concat(pieces,ignore_index=True).sort_values(['a0010','a0030']).reset_index(drop=True)
    assert not d[['a0010','a0030']].duplicated().any()
    assert d.groupby('a0010').a0020.nunique().max()==1
    missing_before=d[TARGETS+EXTRA+DEMO].isin([-999,-666,-888]).sum().to_dict()
    d[TARGETS+EXTRA+DEMO]=d[TARGETS+EXTRA+DEMO].replace([-999,-666,-888],np.nan)
    for t in NUM:assert d[t].dropna().between(-100,100).all(),t
    for t in CAT:assert d[t].dropna().isin([0,1] if t=='c7010' else range(1,6)).all(),t
    bg=pd.read_csv(DATA/'raw'/'ecb.CES_data_background.en.csv',usecols=['a0010','a1020_prec'])
    assert not bg.a0010.duplicated().any()
    d=d.merge(bg,on='a0010',how='left',validate='many_to_one')
    d['case_id']=d.a0010+'-w'+d.a0030.astype(str)
    hold={v:int(hashlib.sha256(v.encode()).hexdigest()[:8],16)%5==0 for v in d.a0010.unique()}
    d['unseen_person']=d.a0010.map(hold)
    d['split']=''
    for split,(lo,hi) in SPLITS.items():d.loc[d.a0030.between(lo,hi),'split']=split
    d['train_eligible_person']=~d.unseen_person
    d['month_sin']=np.sin(2*np.pi*((d.a0030-1)%12)/12)
    d['month_cos']=np.cos(2*np.pi*((d.a0030-1)%12)/12)
    d['wave_trend']=d.a0030
    g=d.groupby('a0010',sort=False)
    lagcols=TARGETS+EXTRA+DEMO+['a0030']
    blocks=[]
    for lag in [1,2,3]:
        z=g[lagcols].shift(lag).add_suffix(f'_lag{lag}')
        blocks.append(z)
    f=pd.concat(blocks,axis=1)
    d=pd.concat([d,f],axis=1)
    d['eligible']=np.logical_and.reduce([d[f'a0030_lag{k}'].eq(d.a0030-k) for k in [1,2,3]])
    g=d.groupby('a0010',sort=False)
    d['history_count']=g.cumcount()
    # Shift FIRST: the current outcome never enters rolling summaries.
    roll=[]
    prior=g[TARGETS+EXTRA].shift(1)
    pg=prior.groupby(d.a0010,sort=False)
    for op in ['mean','std','min','max']:
        z=getattr(pg.rolling(12,min_periods=1),op)().reset_index(level=0,drop=True)
        z=z.reindex(d.index).add_suffix('_hist12_'+op);roll.append(z)
    d=pd.concat([d]+roll,axis=1)
    for t in TARGETS+EXTRA:d[t+'_prior_change']=d[t+'_lag1']-d[t+'_lag2']
    macro=pd.read_parquet(DATA/'macro_panel.parquet')
    d=d.merge(macro,on=['a0020','a0030'],how='left',validate='many_to_one')
    assert d.macro_cutoff.notna().all()
    macrocols=[x for x in macro if x.startswith('macro_') and x!='macro_cutoff']
    # Country codes follow a fixed public codebook, not learned target statistics.
    country_order=['AT','BE','DE','EL','ES','FI','FR','IE','IT','NL','PT']
    for c in country_order:d['country_'+c]=(d.a0020==c).astype('float32')
    basic=['country_'+c for c in country_order]+['a1020_prec','month_sin','month_cos','wave_trend']
    basic += [f'{x}_lag{k}' for k in [1,2,3] for x in TARGETS+DEMO]
    rich=basic+[f'{x}_lag{k}' for k in [1,2,3] for x in EXTRA]
    rich += ['history_count']+[f'{x}_hist12_{op}' for op in ['mean','std','min','max'] for x in TARGETS+EXTRA]
    rich += [x+'_prior_change' for x in TARGETS+EXTRA]
    features={'basic':basic,'rich':rich,'rich_macro':rich+macrocols}
    for name,columns in features.items():
        assert not set(columns)&set(TARGETS+EXTRA+DEMO+['wgt','a0010','unseen_person'])
        assert len(columns)==len(set(columns))
    # Float32 materially reduces unified-memory pressure without rounding survey values.
    for c in d.select_dtypes('float64'):d[c]=d[c].astype('float32')
    d.to_parquet(DATA/'panel.parquet',index=False)
    cases=d[d.eligible].copy()
    cases[['case_id','a0010','a0020','a0030','split','unseen_person','train_eligible_person']].to_parquet(DATA/'split_manifest.parquet',index=False)
    (DATA/'feature_schema.json').write_text(json.dumps(features,indent=2))
    config=dict(seed=SEED,split_waves=SPLITS,person_holdout='sha256(person_id)[:8] modulo 5 equals 0; excluded from supervised training',eligibility='three consecutive preceding monthly records, independent of outcome values',sampling='all eligible rows; no sample cap',selection='tune: prompt/optimizer and checkpoint development; select: final candidate and calibration; retrospective periods not used for fitting',missing='special codes -999/-666/-888 become missing; no future fill; numeric missing outcomes excluded target-wise',history='previous observations only; rolling summary over up to 12 observations with explicit wave range in LLM input',macro='pre-wave conservative cutoff and first publication / revision metadata; see macro_audit.json',survey_vintage='current public CES microdata: earlier survey weights/income quintiles can have retrospective corrections; not claimed original real-time survey vintages')
    (DATA/'split_config.json').write_text(json.dumps(config,indent=2))
    audit=dict(rows=len(d),persons=d.a0010.nunique(),eligible_rows=len(cases),counts=cases.groupby(['split','unseen_person']).size().to_string(),special_missing_counts=missing_before,target_missing=cases.groupby('split')[TARGETS].count().to_dict(),features={k:len(v) for k,v in features.items()},lag_future_violations=int((d.a0030_lag1>=d.a0030).sum()),earliest_monthly_employment_wave=int(d.loc[d.emp_status.notna(),'a0030'].min()))
    (DATA/'dataset_audit.json').write_text(json.dumps(audit,indent=2,default=int))
    hashes={x:hashlib.file_digest((DATA/x).open('rb'),'sha256').hexdigest() for x in ['panel.parquet','split_manifest.parquet','feature_schema.json','split_config.json']}
    (DATA/'dataset_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(audit,indent=2,default=int),flush=True)

if __name__=='__main__':main()
