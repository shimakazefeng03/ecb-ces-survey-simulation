"""First-published HICP, unemployment vintages and dated ECB policy rates."""
from pathlib import Path
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import re
import numpy as np
import pandas as pd
import requests

HERE=Path(__file__).resolve().parent
RAW=HERE/'data/macro_raw'
COUNTRIES=['AT','BE','DE','EL','ES','FI','FR','IE','IT','NL','PT']
API='https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/'

def fetch(task):
    name,url,params=task
    dest=RAW/(name+('.csv' if name=='ecb_deposit' else '.json'))
    if not dest.exists():
        r=requests.get(url,params=params,timeout=(30,180));r.raise_for_status()
        temp=dest.with_suffix('.download');temp.write_bytes(r.content);temp.replace(dest)
        request_url=r.url
    else:
        request_url=requests.Request('GET',url,params=params).prepare().url
    row=dict(name=name,url=request_url,bytes=dest.stat().st_size,
        sha256=hashlib.file_digest(dest.open('rb'),'sha256').hexdigest(),
        acquired_at=dt.datetime.fromtimestamp(dest.stat().st_mtime,dt.timezone.utc).isoformat())
    print('ACQUIRED',name,row['bytes'],flush=True)
    return row

def jsonstat(path):
    j=json.loads(path.read_text());dims=j['id'];shape=j['size'];axes=[]
    for dim in dims:
        ix=j['dimension'][dim]['category']['index']
        if isinstance(ix,list):axes.append(ix)
        else:axes.append([x for x,k in sorted(ix.items(),key=lambda q:q[1])])
    values=j['value'];items=values.items() if isinstance(values,dict) else enumerate(values)
    records=[]
    for ix,v in items:
        if v is None:continue
        pos=np.unravel_index(int(ix),shape)
        rec={d:axes[k][pos[k]] for k,d in enumerate(dims)};rec['value']=v;records.append(rec)
    return pd.DataFrame.from_records(records)

def build():
    old=jsonstat(RAW/'hicp_first_old.json').rename(columns={'coicop':'component'})
    new=jsonstat(RAW/'hicp_first_new.json').rename(columns={'coicop18':'component'})
    h=pd.concat([old,new],ignore_index=True)
    h['component']=h.component.replace({'CP00':'headline','TOTAL':'headline','CP01':'food','NRG':'energy'})
    h['period']=pd.PeriodIndex(h.time,freq='M')
    # Final first publication is normally in the following month. Using the
    # first day of the second following month deliberately excludes that
    # month's flash estimate and avoids needing an individual interview date.
    h['available_by']=(h.period+2).dt.to_timestamp()
    assert not h[['geo','period','component']].duplicated().any()
    u=pd.concat([jsonstat(RAW/'unemployment_old_vintages.json'),jsonstat(RAW/'unemployment_vintages.json')],ignore_index=True)
    u['revision_date']=pd.to_datetime(u.revdate)
    u['period']=pd.PeriodIndex(u.time,freq='M')
    u=u.sort_values('revision_date').drop_duplicates(['geo','period','revision_date'],keep='last')
    d=pd.read_csv(RAW/'ecb_deposit.csv',usecols=['TIME_PERIOD','OBS_VALUE'])
    d['date']=pd.to_datetime(d.TIME_PERIOD);d=d.sort_values('date')
    rows=[];lineage=[]
    guide=(HERE.parents[1]/'sources/ecb/ecb.CES_microdata_guide.en.pdf.txt').read_text()
    section=guide[guide.index('Running number, starting from 4'):]
    section=section[:section.index('WGT: Blended weight monthly')]
    # The source has a missing slash for wave 53; preserve an explicit repair.
    section=section.replace('2/52024','2/5/2024')
    starts={int(w):pd.to_datetime(date,dayfirst=True) for w,date in re.findall(r'\n(\d+)\s*\nIf date\s*=\s*(\d+/\d+/\d+)',section)}
    for wave in range(4,79):
        nominal=pd.Timestamp(year=2020+(wave-1)//12,month=(wave-1)%12+1,day=1)
        # Forecast at midnight on the last day of the PRECEDING calendar
        # month. This also precedes the exceptional 30 April 2025 opening.
        cutoff=nominal-pd.Timedelta(days=1)
        if wave in starts:assert cutoff<=starts[wave]
        policy=d[d.date<cutoff]
        dr=policy.iloc[-1]
        prev=d[d.date<cutoff-pd.DateOffset(months=3)].iloc[-1]
        for country in COUNTRIES:
            rec={'a0020':country,'a0030':wave,'macro_cutoff':cutoff.strftime('%Y-%m-%d'),
                 'macro_deposit_rate':float(dr.OBS_VALUE),'macro_deposit_change3':float(dr.OBS_VALUE-prev.OBS_VALUE)}
            lineage.append(dict(country=country,wave=wave,factor='deposit',reference_date=dr.date.strftime('%Y-%m-%d'),information_date=dr.date.strftime('%Y-%m-%d'),cutoff=rec['macro_cutoff'],vintage='dated_policy_rate'))
            for component in ['headline','food','energy']:
                v=h[(h.geo==country)&(h.component==component)&(h.available_by<=cutoff)].sort_values('period')
                if len(v):
                    last=v.iloc[-1];values=v.set_index('period').value
                    rec['macro_'+component]=float(last.value)
                    for k in [1,3]:rec[f'macro_{component}_change{k}']=float(last.value-values.get(last.period-k,np.nan))
                    rec['macro_'+component+'_age_months']=nominal.to_period('M').ordinal-last.period.ordinal
                    lineage.append(dict(country=country,wave=wave,factor=component,reference_date=str(last.period),information_date=last.available_by.strftime('%Y-%m-%d'),cutoff=rec['macro_cutoff'],vintage='first_published_final_conservative_availability'))
            v=u[(u.geo==country)&(u.revision_date<cutoff)&(u.period<cutoff.to_period('M'))]
            v=v.sort_values('revision_date').drop_duplicates('period',keep='last').sort_values('period')
            if len(v):
                last=v.iloc[-1];values=v.set_index('period').value
                rec['macro_unemployment']=float(last.value)
                for k in [1,3]:rec[f'macro_unemployment_change{k}']=float(last.value-values.get(last.period-k,np.nan))
                rec['macro_unemployment_age_months']=nominal.to_period('M').ordinal-last.period.ordinal
                lineage.append(dict(country=country,wave=wave,factor='unemployment',reference_date=str(last.period),information_date=last.revision_date.strftime('%Y-%m-%d'),cutoff=rec['macro_cutoff'],vintage='eurostat_revision_date'))
            rows.append(rec)
    out=HERE/'data';pd.DataFrame(rows).to_parquet(out/'macro_panel.parquet',index=False)
    line=pd.DataFrame(lineage)
    assert (pd.to_datetime(line.information_date)<=pd.to_datetime(line.cutoff)).all()
    line.to_csv(out/'macro_lineage.csv',index=False)
    audit={'rows':len(rows),'countries':COUNTRIES,'waves':[4,78],
           'missing_by_factor':pd.DataFrame(rows).isna().sum().to_dict(),
           'hicp_availability_rule':'First-published final observations enter at reference month plus two months, day 1. This is conservative rather than a row-specific release timestamp.',
           'unemployment_vintage_rule':'For every reference month use only revisions dated strictly before the declared cutoff.',
           'forecast_cutoff':'00:00 on the final calendar day before the nominal survey month; no same-day policy/unemployment releases.',
           'fieldwork_start_verification':{'guide_waves':sorted(starts),'verified_pre_opening':True,'later_waves':'Exact dates absent from the cached guide. Cutoff is a conservative inference from the documented first-Thursday schedule; not an independently verified interview-date boundary.'},
           'hicp_metadata_source':'https://ec.europa.eu/eurostat/cache/metadata/EN/prc_hicp_esms.htm',
           'release_lag_note':'Vintage archive publication lag does not change original recorded information dates; this is a retrospective reconstruction, not evidence of a live forecast.',
           'hicp_2026_break':'ECOICOP classification transition: original first-published old series through 2025; new series from 2026. Broad total/food/energy components retained.',
           'lineage_future_information_violations':0}
    (out/'macro_audit.json').write_text(json.dumps(audit,indent=2))
    print('MACRO_PANEL',json.dumps(audit),flush=True)

def main():
    RAW.mkdir(parents=True,exist_ok=True)
    geos=[('geo',x) for x in COUNTRIES]
    common=[('lang','en'),('unit','RCH_A'),('release','FIN')]+geos
    labor=[('lang','en'),('unit','PC_ACT'),('s_adj','SA'),('sex','T'),('age','TOTAL'),('sinceTimePeriod','2019-01')]+geos
    tasks=[
      ('hicp_first_old',API+'prc_hicp_fp',common+[('sinceTimePeriod','2019-01')]+[('coicop',x) for x in ['CP00','CP01','NRG']]),
      ('hicp_first_new',API+'prc_hicp_fpd',common+[('sinceTimePeriod','2026-01')]+[('coicop18',x) for x in ['TOTAL','CP01','NRG']]),
      ('unemployment_old_vintages',API+'ei_lm_m_vtgfix',labor),
      ('unemployment_vintages',API+'ei_lm_m_vtg',labor),
      ('ecb_deposit','https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV',[('startPeriod','2019-01-01'),('format','csvdata')])]
    with cf.ThreadPoolExecutor(max_workers=3) as ex:manifest=list(ex.map(fetch,tasks))
    (RAW.parent/'macro_manifest.json').write_text(json.dumps(manifest,indent=2))
    build()

if __name__=='__main__':main()
