"""Read-only source audit. Extract metadata and same-respondent candidate fields.
Never modifies source CSVs, models, inference or historical experiments.
"""
import argparse,csv,hashlib,html,json,re,shutil,zipfile
from collections import Counter
from pathlib import Path

def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def plain(text):return html.unescape(re.sub('<[^>]+>','',re.sub(r'<br\s*/?>','\n',text,flags=re.I))).replace('\xa0',' ').strip()

def catalog(root,out):
    if out.exists():raise RuntimeError('New audit directory required.')
    out.mkdir(parents=True)
    book=root/'data/baseline_survey_dataset/raw/codebook/USCODE22_LLCP_102523.HTML'
    text=book.read_text(encoding='cp1252');matches=list(re.finditer(r'<td[^>]*class="l m linecontent"[^>]*>(Label:.*?)</td>',text,re.S));fields={}
    for i,m in enumerate(matches):
        desc=plain(m.group(1));meta=dict(re.findall(r'^([^:\n]+):\s*(.*)$',desc,re.M))
        name=meta.get('SAS Variable Name');position=meta.get('Column','')
        if not name or not re.fullmatch(r'\d+(?:-\d+)?',position):continue
        start=int(position.split('-')[0]);end=int(position.split('-')[-1])
        body=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)];values=[]
        for row in re.findall(r'<tr[^>]*>(.*?)</tr>',body,re.S):
            cells=[plain(x) for x in re.findall(r'<td[^>]*>(.*?)</td>',row,re.S)]
            if len(cells)>=2:values.append(dict(code=cells[0],description=cells[1]))
        fields[name]=dict(start=start,length=end-start+1,metadata=meta,values=values)
    save(out/'codebook_catalog.json',fields)
    protected=[p for p in (root/'models/current_models').rglob('*') if p.is_file()]+[root/p for p in ['models/active_model.json','data/expanded_survey_dataset/respondents.csv','data/baseline_survey_dataset/raw/LLCP2022ASC.zip','training/train_expanded_models.py','src/brfss_inference.py','src/ensemble_policy.py']]+[book]
    save(out/'original_hashes.json',{str(p.relative_to(root)):digest(p) for p in protected})
    print('CATALOG',len(fields),'fields; original hashes recorded.')
    names=['MENTHLTH','POORHLTH','LSATISFY','ADDEPEV3','PERSDOC3','CHECKUP1','PRIMINSR','CHILDREN','RENTHOM1','VETERAN3','CVDINFR4','CVDCRHD4','CVDSTRK3','ASTHMA3','ASTHNOW','CHCCOPD3','CHCKDNY2','DIABETE4','HAVARTH4','DEAF','BLIND','DECIDE','DIFFWALK','DIFFDRES','DIFFALON','SMOKE100','SMOKDAY2','USENOW3','ECIGNOW2','ALCDAY4','AVEDRNK3','DRNK3GE5','MAXDRNKS','SDHFOOD','SDHFOOD1','SDHSTRE1']
    for name in names:
        f=fields.get(name)
        print(name,json.dumps(f) if f else 'NOT IN CODEBOOK')

def extract(root,out):
    fields=json.loads((out/'codebook_catalog.json').read_text());spec=json.loads((out/'candidate_spec.json').read_text())
    for name,h in json.loads((out/'original_hashes.json').read_text()).items():assert digest(root/name)==h,name
    with (root/'data/expanded_survey_dataset/respondents.csv').open(newline='') as stream:source=list(csv.DictReader(stream))
    rows={r['respondent_id']:r for r in source};assert len(rows)==len(source)
    names=list(spec);counts={n:Counter() for n in names};training={n:Counter() for n in names};matches=set();mismatch=0
    dest=out/'same_respondent_raw_candidates.csv'
    if dest.exists():raise RuntimeError('Do not overwrite completed extraction.')
    with dest.open('w',newline='') as output,zipfile.ZipFile(root/'data/baseline_survey_dataset/raw/LLCP2022ASC.zip') as archive:
        writer=csv.DictWriter(output,fieldnames=['respondent_id','split','profile_group']+names);writer.writeheader()
        with archive.open(archive.namelist()[0]) as stream:
            for line_number,line in enumerate(stream,1):
                assert len(line.rstrip(b'\r\n'))==2052
                rid='2022_'+line[:2].decode()+'_'+line[35:45].decode().strip()
                if rid not in rows:continue
                assert rid not in matches;matches.add(rid);old=rows[rid]
                assert int(old['source_row'])==line_number
                f=fields['SDHSTRE1'];stress=int(line[f['start']-1:f['start']-1+f['length']].strip())
                assert stress==int(old['stress_raw']) and int(old['target'])==(0 if stress>=4 else 1 if stress==3 else 2)
                new=dict(respondent_id=rid,split=old['split'],profile_group=old['profile_group'])
                for n in names:
                    f=fields[n];value=line[f['start']-1:f['start']-1+f['length']].decode().strip();new[n]=value;counts[n][value]+=1
                    if old['split']=='train':training[n][value]+=1
                writer.writerow(new)
    assert matches==set(rows)
    save(out/'extraction.json',dict(source_records=line_number,matched=len(matches),unique_respondents=len(rows),raw_value_counts=counts,training_raw_value_counts=training,candidate_sha256=digest(dest),no_target_mismatches=True))
    print('EXTRACTED',len(matches),'same respondents; all source rows, targets and IDs verified.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['catalog','extract']);p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();globals()[a.phase](Path(a.project),Path(a.output))
