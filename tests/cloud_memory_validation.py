"""Exercise the actual Gunicorn API inside a memory-limited Linux cgroup."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

parser = argparse.ArgumentParser()
parser.add_argument('--project', required=True)
parser.add_argument('--compact', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--installed', action='store_true')
args = parser.parse_args()
root = Path(args.project)
candidate = Path(__file__).resolve().parents[1]
token = secrets.token_urlsafe(32)
env = dict(os.environ, INFERENCE_API_KEY=token, BRFSS_COMPACT_FOREST_DIR=args.compact,
           VALIDATION_PROJECT=str(root), PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
command = [sys.executable, '-m', 'gunicorn', '--bind', f'127.0.0.1:{port}',
           '--workers', '1', '--threads', '4', '--timeout', '120']
if args.installed:
    command += ['deploy.backend:application']
    cwd = root
else:
    command += ['--pythonpath', str(candidate / 'tests'), 'candidate_backend:application']
    cwd = candidate
proc = subprocess.Popen(command, env=env, cwd=cwd)

def request(path, payload=None, authorized=True):
    headers = {'Content-Type':'application/json'}
    if authorized:
        headers['Authorization'] = 'Bearer ' + token
    req = Request(f'http://127.0.0.1:{port}{path}', headers=headers,
                  data=json.dumps(payload).encode() if payload is not None else None)
    try:
        with urlopen(req, timeout=30) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)

try:
    deadline = time.monotonic() + 120
    while True:
        if proc.poll() is not None:
            raise RuntimeError('API exited before becoming ready')
        try:
            if request('/healthz', authorized=False)[0] == 200:
                break
        except (URLError, TimeoutError, ConnectionError):
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('API readiness timed out')
        time.sleep(0.2)
    assert request('/api/health', authorized=False)[0] == 401
    assert request('/api/health')[1]['models_loaded']
    assert request('/api/schema')[1]['input_count'] == 19
    assert request('/api/benchmarks')[0] == 200
    assert request('/api/predict', {})[0] == 400
    schema = json.loads((root/'models/current_models/schema.json').read_text())
    payloads = []
    with (root/'data/expanded_survey_dataset/respondents.csv').open(newline='') as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index % 357 == 0 and len(payloads) < 512:
                payloads.append({k:float(row[k]) for k in schema['numeric']+schema['categorical']})
    start = time.monotonic()
    results = []
    for payload in payloads:
        status, result = request('/api/predict', payload)
        assert status == 200 and len(result['models']) == 6
        results.append(result)
    # Concurrent calls must remain bounded; 503 is explicit backpressure, not
    # a fake prediction or an unbounded queue of heavy inference allocations.
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: request('/api/predict', payloads[0]), range(16)))
    assert all(status in (200,503) for status,_ in responses)
    assert request('/healthz', authorized=False)[0] == 200
    cg_name = Path('/proc/self/cgroup').read_text().strip().split('::',1)[1]
    cg = Path('/sys/fs/cgroup') / cg_name.lstrip('/')
    report = {'requests':len(payloads), 'model_predictions':len(payloads)*6,
              'seconds':time.monotonic()-start,
              'memory_peak_bytes':int((cg/'memory.peak').read_text()),
              'memory_current_bytes':int((cg/'memory.current').read_text()),
              'memory_stat':(cg/'memory.stat').read_text().strip(),
              'memory_limit':(cg/'memory.max').read_text().strip(),
              'swap_limit':(cg/'memory.swap.max').read_text().strip(),
              'memory_events':(cg/'memory.events').read_text().strip(),
              'concurrent_statuses':[status for status,_ in responses]}
    assert report['memory_limit'] == '536870912'
    assert 'oom_kill 0' in report['memory_events']
    Path(args.output).write_text(json.dumps({'summary':report, 'results':results},indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
finally:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
