"""Execute saved report-only notebook cells using the project's existing ML runtime."""
import argparse,contextlib,io,json,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('path');a=p.parse_args();path=Path(a.path).resolve();os.chdir(path.parent);nb=json.loads(path.read_text());env={};count=0
for cell in nb['cells']:
    if cell['cell_type']!='code':continue
    count+=1;buf=io.StringIO()
    with contextlib.redirect_stdout(buf):exec(compile(''.join(cell['source']),str(path),'exec'),env)
    cell['execution_count']=count;cell['outputs']=[dict(output_type='stream',name='stdout',text=buf.getvalue().splitlines(True))]
path.write_text(json.dumps(nb,indent=2),encoding='utf-8');print('Executed',count,'report-only cells successfully.')
