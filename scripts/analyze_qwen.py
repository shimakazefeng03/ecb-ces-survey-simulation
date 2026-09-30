import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from scipy.spatial.distance import jensenshannon
from prepare_data import ROOT

def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def main():
    rows=[];predictions=[]
    for mode in ['base','hard','soft']:
        path=ROOT/'results/qwen'/mode
        step=0 if mode=='base' else json.loads((path/'selection.json').read_text())['selected_step']
        v=read(path/f'valid_{step}.jsonl');test=read(path/f'test_{step}.jsonl')
        vl=np.array([r['logits'] for r in v]);vy=np.array([r['label'] for r in v])
        fit=minimize_scalar(lambda logt:-np.log(softmax(vl/np.exp(logt),axis=1)[np.arange(len(vy)),vy].clip(1e-12)).mean(),bounds=(-4,4),method='bounded')
        logits=np.array([r['logits'] for r in test]);y=np.array([r['label'] for r in test]);truth=np.eye(5)[y];ids=[r['id'] for r in test]
        for name,temp in [('raw',1),('temperature',float(np.exp(fit.x)))]:
            p=softmax(logits/temp,axis=1);hard=p.argmax(axis=1);freq=np.bincount(hard,minlength=5)/len(hard)
            out={'model':mode,'calibration':name,'n':len(y),'temperature':temp,'selected_step':step,'accuracy':float((hard==y).mean()),
                 'ordinal_mae':float(abs(hard-y).mean()),'nll':float(-np.log(p[np.arange(len(y)),y].clip(1e-12)).mean()),
                 'brier':float(((p-truth)**2).sum(axis=1).mean()),'jsd_probability_mean':float(jensenshannon(p.mean(axis=0),truth.mean(axis=0),base=2)**2),
                 'jsd_argmax':float(jensenshannon(freq,truth.mean(axis=0),base=2)**2)}
            rows.append(out)
            for cid,pp,yy in zip(ids,p,y):predictions.append({'case_id':cid,'mode':mode,'calibration':name,'label':int(yy),'probabilities':pp.tolist()})
    pd.DataFrame(rows).to_csv(ROOT/'results/qwen_metrics.csv',index=False)
    (ROOT/'results/qwen_predictions.json').write_text(json.dumps(predictions))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
