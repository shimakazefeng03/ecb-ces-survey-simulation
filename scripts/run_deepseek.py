"""Paired prompt experiment. Append-only raw responses allow restart without replay."""
import concurrent.futures as cf
from datetime import datetime,timezone
import json,re,time,argparse
import numpy as np
import pandas as pd
from prepare_data import ROOT,TARGETS,NUM,SEED
from deepseek_client import call,key

COUNTRIES={'AT':'Austria','BE':'Belgium','DE':'Germany','EL':'Greece','ES':'Spain','FI':'Finland','FR':'France','IE':'Ireland','IT':'Italy','NL':'Netherlands','PT':'Portugal'}
QUESTIONS='''Answer these Consumer Expectations Survey items as this respondent, from their subjective viewpoint:
c1120: By what percentage do you expect prices in general in your country to change over the next 12 months? Increase positive, decrease negative; percent, -100 to 100.
c1220: By what percentage do you expect prices in general in your country to change over the 12-month period between two and three years from now? Annual rate, NOT cumulative three-year change; percent, -100 to 100.
c6120: Compared with the past 12 months, by what percentage do you expect your household's total nominal spending to change over the next 12 months? -100 to 100, or null if you do not know.
c3010: Compared with 12 months ago, your household financial situation now is: 1 much worse, 2 somewhat worse, 3 about the same, 4 somewhat better, 5 much better.
c3110: Compared with now, your household financial situation in 12 months will be: 1 much worse, 2 somewhat worse, 3 about the same, 4 somewhat better, 5 much better.
c7010: Think about your available financial resources, including access to credit, savings, loans from relatives or friends, etc. If you had to make an unexpected payment equal to one month of your household income, would you have sufficient financial resources to pay for the entire amount? 0 no, 1 yes.
Return ONLY a JSON object with exactly these six keys. Use one numeric answer per question (null is allowed only for c6120). Do not explain. Do not answer as an economist predicting actual outcomes. The target is this person's subjective survey answer.'''

def persona(row):
    ages={1:'18-34',2:'35-49',3:'50-70',4:'71 or older'}
    return {'country':COUNTRIES[row.a0020],'age_group':ages[int(row.a1010_age_prec_lag1)],
            'public_survey_sex_category':{1:'male',2:'female'}[int(row.a1020_prec)],
            'household_income_quintile_within_country':int(row.b7040_quintile_lag1)}

def prompt(row,history=None,lags=(3,2,1)):
    year=2020+(int(row.a0030)-1)//12;month=(int(row.a0030)-1)%12+1
    p=f'Survey month: {year}-{month:02d}. Public anonymized respondent profile recorded before this month: '+json.dumps(persona(row))+'.\n'
    if history is not None:
        records=[]
        for lag in lags:
            h={'months_before_survey':lag}
            for t in TARGETS:
                v=history[f'{t}_lag{lag}'];h[t]=None if pd.isna(v) else float(v)
            records.append(h)
        p+='This respondent previously gave the following answers. Treat them as past answers, not current answers. Change is possible. The keys refer to the same questions listed below.\n'+json.dumps(records)+'\n'
    return p+QUESTIONS

def parse(text):
    text=re.sub(r'^```(?:json)?\s*|\s*```$','',text.strip())
    d=json.loads(text)
    if set(d)!=set(TARGETS):raise ValueError('wrong keys')
    for t in TARGETS:
        v=d[t]
        if v is None and t=='c6120':continue
        if not isinstance(v,(float,int)) or isinstance(v,bool):raise ValueError('non numeric')
        if t in NUM and not(-100<=v<=100):raise ValueError('range')
        if t in ['c3010','c3110'] and v not in [1,2,3,4,5]:raise ValueError('category')
        if t=='c7010' and v not in [0,1]:raise ValueError('binary')
    return d

def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=12);a.add_argument('--repeat',action='store_true');a.add_argument('--ablation',action='store_true');args=a.parse_args()
    sample=pd.read_pickle(ROOT/'data/processed/test_sample.pkl')
    # Cyclic random shift is a derangement within each country-month (never own history).
    rng=np.random.default_rng(SEED);shuffled={};matched={}
    for _,g in sample.groupby(['a0030','a0020']):
        ix=rng.permutation(g.index);shift=int(rng.integers(1,len(ix)))
        shuffled.update(dict(zip(ix,np.roll(ix,shift))))
        from scipy.optimize import linear_sum_assignment
        vals=g[['a1010_age_prec_lag1','b7040_quintile_lag1','a1020_prec']].to_numpy()
        cost=np.abs(vals[:,None,:]-vals[None,:,:]).sum(axis=2).astype(float)
        np.fill_diagonal(cost,1e6)
        src,dst=linear_sum_assignment(cost);matched.update(dict(zip(g.index[src],g.index[dst])))
    jobs=[]
    for i,row in sample.iterrows():
        arms=['demo','history','shuffled_history'] if not args.repeat else (['history_repeat'] if i%5==0 else [])
        if args.ablation:arms=['history1','matched_history','reversed_history']
        for arm in arms:
            hist=None if arm=='demo' else (sample.loc[shuffled[i]] if arm=='shuffled_history' else row)
            if arm=='matched_history':hist=sample.loc[matched[i]]
            if arm=='reversed_history':
                hist=row.copy()
                for t in TARGETS:
                    hist[f'{t}_lag1']=row[f'{t}_lag3'];hist[f'{t}_lag3']=row[f'{t}_lag1']
            jobs.append({'case_id':row.case_id,'arm':arm,'prompt':prompt(row,hist,lags=(1,) if arm=='history1' else (3,2,1))})
    rng.shuffle(jobs)
    path=ROOT/'results/deepseek_raw.jsonl'
    done={}
    if path.exists():
        for line in path.read_text().splitlines():
            z=json.loads(line)
            if 'answers' in z:done[(z['case_id'],z['arm'])]=z
    jobs=[z for z in jobs if (z['case_id'],z['arm']) not in done]
    key();print('pending',len(jobs),flush=True)
    def worker(job):
        job=dict(job);job['started_at']=datetime.now(timezone.utc).isoformat()
        try:
            result=call([{'role':'system','content':'You simulate a single anonymized survey respondent. Follow the question definitions exactly.'},{'role':'user','content':job['prompt']}],temperature=.7,max_tokens=800,thinking=False)
            job['response']=result;job['answers']=parse(result['choices'][0]['message']['content'])
        except Exception as e:job['error']=str(e)
        return job
    with path.open('a') as out,cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(worker,job) for job in jobs]
        for n,future in enumerate(cf.as_completed(pending),1):
            z=future.result()
            out.write(json.dumps(z,ensure_ascii=False)+'\n');out.flush()
            if n%30==0 or 'error' in z:print(n,'/',len(jobs),z.get('error','ok'),flush=True)

if __name__=='__main__':main()
