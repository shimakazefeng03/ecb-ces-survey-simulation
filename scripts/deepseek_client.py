"""Small authenticated client. Secrets remain in process memory, never logs/files."""
import json, subprocess, time
from pathlib import Path
import requests

MODEL = 'deepseek-flash'
BASE = 'https://api.deepseek.com'
_key = None

def key():
    global _key
    if _key is None:
        r = subprocess.run(['security','find-generic-password','-s','deepseek-api-key','-w'], capture_output=True,text=True)
        if r.returncode: raise RuntimeError('DeepSeek Keychain lookup failed')
        _key=r.stdout.strip()
    return _key

def call(messages, *, temperature=0.7, max_tokens=2048, thinking=False):
    payload={'model':MODEL,'messages':messages,'temperature':temperature,'max_tokens':max_tokens,'stream':False,
             'thinking':{'type':'enabled' if thinking else 'disabled'}}
    started=time.time()
    for attempt in range(4):
        r=requests.post(BASE+'/chat/completions',headers={'Authorization':'Bearer '+key()},json=payload,timeout=180)
        if r.status_code in [429,500,502,503,504] and attempt<3:
            time.sleep(2**attempt);continue
        if not r.ok: raise RuntimeError(f'DeepSeek HTTP {r.status_code}: {r.text[:300]}')
        obj=r.json();obj['elapsed_seconds']=time.time()-started
        return obj
    raise RuntimeError('DeepSeek retries exhausted')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('prompt');p.add_argument('output');p.add_argument('--thinking',action='store_true');a=p.parse_args()
    msg=Path(a.prompt).read_text()
    obj=call([{'role':'system','content':'You are a careful empirical research reviewer. Distinguish evidence, assumptions and unverified claims. Do not invent citations.'},{'role':'user','content':msg}],temperature=.3,max_tokens=12000,thinking=a.thinking)
    Path(a.output).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
    Path(a.output).with_suffix('.md').write_text(obj['choices'][0]['message']['content'])
    print(json.dumps({'model':obj.get('model'),'usage':obj.get('usage'),'seconds':obj['elapsed_seconds'],'output':a.output}))
