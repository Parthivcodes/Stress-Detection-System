"""Recheck flagged training responses against original fixed-width survey records."""
import argparse,csv,json,zipfile
from pathlib import Path

def run(root,out):
    names=['ambiguity_review_sample.csv','cleanlab_raw_class_weighted_sample.csv','cleanlab_approximate_weight_reversal_sample.csv'];rows={}
    for name in names:
        with (out/name).open(newline='') as f:
            for r in csv.DictReader(f):rows[r['respondent_id']]=r
    # First file has no source_row; source provenance supplied by original table.
    with (root/'data/expanded_survey_dataset/respondents.csv').open(newline='') as f:
        source={r['respondent_id']:r for r in csv.DictReader(f) if r['respondent_id'] in rows}
    found=set()
    with zipfile.ZipFile(root/'data/baseline_survey_dataset/raw/LLCP2022ASC.zip') as z:
        with z.open(z.namelist()[0]) as stream:
            for lineno,line in enumerate(stream,1):
                rid='2022_'+line[:2].decode()+'_'+line[35:45].decode().strip()
                if rid not in source:continue
                r=source[rid];stress=int(line[369:370]);assert int(r['source_row'])==lineno and stress==int(r['stress_raw'])
                assert int(r['target'])==(0 if stress in [4,5] else 1 if stress==3 else 2);found.add(rid)
    assert found==set(rows)
    result=dict(reviewed_unique_flagged_records=len(rows),all_recorded_answers_and_mapping_match_source=True,verified_label_errors=0,interpretation='No transcription/target-mapping error found in this sample. This cannot verify subjective truth of survey answers. Several highly flagged older respondents report high stress despite otherwise favorable measured predictors. Such combinations are possible and do not justify changing their labels.',action='Retain all rows and labels. No source evidence supports automated relabeling or removal.')
    (out/'source_label_review.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--project',required=True);p.add_argument('--output',required=True);a=p.parse_args();run(Path(a.project),Path(a.output))
