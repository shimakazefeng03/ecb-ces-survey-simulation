from prepare_data import ROOT,TARGETS,SEED
from run_deepseek import persona
import json
import pandas as pd
import numpy as np

def prompt(r):
    h=[]
    for lag in [3,2,1]:
        h.append([None if pd.isna(r[f'{t}_lag{lag}']) else float(r[f'{t}_lag{lag}']) for t in TARGETS])
    return ('Predict this anonymized respondent\'s subjective survey answer, not objective economic outcomes. Profile: '+json.dumps(persona(r))+'. '
        'Past answers, oldest to latest month: '+json.dumps(h)+'. Columns: 1-year inflation %, annual inflation between years 2 and 3 %, expected nominal spending growth %, past financial situation (1 much worse, 2 somewhat worse, 3 same, 4 somewhat better, 5 much better), expected financial situation (same 1-5 scale), ability to meet a one-month-income emergency using all resources including credit/loans (0 no, 1 yes). '
        'Question for the next survey month: How do you expect your household financial situation to change over the next 12 months compared with now? '
        'Answer one digit only: 1 much worse, 2 somewhat worse, 3 about the same, 4 somewhat better, 5 much better.')

def main():
    d=pd.read_pickle(ROOT/'data/processed/lagged.pkl')
    train=d[d.eligible & d.a0030.le(72) & ~d.unseen_person]
    valid=d[d.eligible & d.a0030.between(70,72) & d.unseen_person].sample(220,random_state=SEED)
    test=pd.read_pickle(ROOT/'data/processed/test_sample.pkl')
    train=train.sample(1024,random_state=SEED)
    counts=train.groupby(['a0020','c3110_lag1','c3110']).size()
    def target(r):
        c=np.array([counts.get((r.a0020,r.c3110_lag1,j),0)+1 for j in range(1,6)])
        return (c/c.sum()).tolist()
    for name,frame in [('train',train),('valid',valid),('test',test)]:
        records=[]
        for i,r in frame.iterrows():
            records.append({'id':r.get('case_id',name+str(i)),'prompt':prompt(r),'label':int(r.c3110)-1,'soft_target':target(r),
                            'wave':int(r.a0030),'unseen_person':bool(r.unseen_person)})
        (ROOT/f'data/processed/qwen_{name}.json').write_text(json.dumps(records,ensure_ascii=False))
    print('Qwen train1024 / validation220 / test660; hard labels and historical transition soft labels')

if __name__=='__main__':main()
