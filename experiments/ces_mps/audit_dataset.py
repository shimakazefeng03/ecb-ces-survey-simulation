"""Independent row-level feature reconstruction and temporal boundary checks."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from build_dataset import HERE,DATA,TARGETS,NUM,EXTRA


def main():
    d=pd.read_parquet(DATA/'panel.parquet').sort_values(['a0010','a0030']).reset_index(drop=True)
    schema=json.loads((DATA/'feature_schema.json').read_text())
    saved=json.loads((DATA/'dataset_hashes.json').read_text())
    for f,h in saved.items():assert hashlib.file_digest((DATA/f).open('rb'),'sha256').hexdigest()==h,f
    assert not d.case_id.duplicated().any()
    for feature_list in schema.values():assert not set(feature_list)&set(TARGETS+EXTRA+['wgt','case_id','a0010'])
    train=d[d.eligible&d.train_eligible_person&d.split.eq('train')]
    assert train.a0030.max()<d[d.split.eq('tune')].a0030.min()
    assert d[d.split.eq('tune')].a0030.max()<d[d.split.eq('select')].a0030.min()
    assert d[d.split.eq('select')].a0030.max()<d[d.split.eq('retrospective_2025')].a0030.min()
    assert not set(train.a0010)&set(d.loc[d.unseen_person,'a0010'])
    # All rows: shifts independently checked by keyed self-join. At least
    # 3-history rows are the actual paired population used by every method.
    eligible=d[d.eligible]
    key=d.set_index(['a0010','a0030'])
    tested=0
    for lag in [1,2,3]:
        idx=pd.MultiIndex.from_arrays([eligible.a0010,eligible.a0030-lag])
        for col in TARGETS+EXTRA:
            a=eligible[f'{col}_lag{lag}'].to_numpy();b=key[col].reindex(idx).to_numpy()
            np.testing.assert_allclose(a,b,equal_nan=True);tested+=len(a)
    # Row-level independent historical summaries; evenly-spaced, label-blind
    # audit records cover the whole panel, not an empirical evaluation sample.
    audit_ids=sorted(d.a0010.unique())[::101]
    audit_rows=0
    for _,person in d[d.a0010.isin(audit_ids)].groupby('a0010',sort=False):
        for i,row in enumerate(person.itertuples(index=False)):
            if not row.eligible:continue
            before=person.iloc[max(0,i-12):i]
            for col in TARGETS+EXTRA:
                vals=before[col]
                for op in ['mean','std','min','max']:
                    expected=getattr(vals,op)()
                    actual=getattr(row,f'{col}_hist12_{op}')
                    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-5,equal_nan=True)
            audit_rows+=1
    line=pd.read_csv(DATA/'macro_lineage.csv')
    assert (pd.to_datetime(line.information_date)<=pd.to_datetime(line.cutoff)).all()
    observed=line[line.factor.isin(['deposit','unemployment'])]
    assert (pd.to_datetime(observed.information_date)<pd.to_datetime(observed.cutoff)).all()
    result=dict(status='pass',all_eligible_lag_values_checked=tested,rolling_rows_independently_reconstructed=audit_rows,macro_lineage_rows=len(line),split_hashes_verified=True,supervised_train_vs_forced_heldout_ids_disjoint=True,limit='Row reconstruction checks implemented features. Macro availability still depends on documented release policy and stated date-inference limits; latest CES historical vintages may contain publisher revisions.')
    (HERE/'results/data_integrity_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
