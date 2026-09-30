"""Download official yearly CES files without altering previous experiment inputs."""
from pathlib import Path
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import shutil
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/ces_mps/data/raw'
BASE = 'https://www.ecb.europa.eu/stats/ecb_surveys/consumer_exp_survey/shared/pdf/'

def acquire(year):
    name = f'ecb.CES_data_{year}_monthly.en.csv'
    dst = OUT / name
    source = ROOT / 'data/raw' / name
    origin = 'existing_local_snapshot' if source.exists() else 'official_download'
    if not dst.exists():
        if source.exists():
            shutil.copy2(source, dst)
        else:
            tmp = dst.with_suffix('.download')
            with requests.get(BASE + name, stream=True, timeout=(30, 120)) as r:
                r.raise_for_status()
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            with tmp.open('rb') as f:
                if not f.readline().lower().startswith(b'a0010,'):
                    raise ValueError(f'{name}: unexpected non-CSV content')
            tmp.replace(dst)
    sha = hashlib.file_digest(dst.open('rb'), 'sha256').hexdigest()
    result = dict(year=year, path=str(dst.relative_to(ROOT)), url=BASE+name,
                  bytes=dst.stat().st_size, sha256=sha, origin=origin)
    print(json.dumps(result), flush=True)
    return result

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        rows = list(ex.map(acquire, range(2020,2027)))
    bg = ROOT / 'data/raw/ecb.CES_data_background.en.csv'
    shutil.copy2(bg, OUT / bg.name)
    (OUT.parent/'ces_manifest.json').write_text(json.dumps(dict(
        checked_at=dt.datetime.now(dt.timezone.utc).isoformat(), files=rows,
        latest_official_microdata='2026-Q2', background_sha256=hashlib.file_digest(bg.open('rb'),'sha256').hexdigest(),
        source='ECB Consumer Expectations Survey'),indent=2))

if __name__ == '__main__': main()
