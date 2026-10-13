"""Run one fictional case and record a separate, explicit human approval."""
import argparse,asyncio,json
from pathlib import Path
from agentic.workflow import run,invoke
BASE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--case',default=str(BASE/'demo_case.json'),help='JSON fictional assessment')
    p.add_argument('--actor',default='demo-worker')
    p.add_argument('--review-case',help='Existing case ID to review')
    p.add_argument('--reviewer',help='Name of human reviewer (only for --review-case)')
    p.add_argument('--decision',choices=['approved_demo_record','needs_more_information','rejected'])
    p.add_argument('--note',default='')
    p.add_argument('--audit-case',help='Show full audit for case ID')
    args=p.parse_args()
    if args.audit_case:
        result=asyncio.run(invoke('get_case_audit',{'case_id':args.audit_case}))
    elif args.review_case:
        if not args.reviewer or not args.decision:p.error('--review-case requires --reviewer and --decision')
        result=asyncio.run(invoke('record_clinician_review',{'case_id':args.review_case,'reviewer':args.reviewer,'decision':args.decision,'note':args.note}))
    else:
        assessment=json.loads(Path(args.case).read_text(encoding='utf-8'))
        result=asyncio.run(run(assessment,args.actor))
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
