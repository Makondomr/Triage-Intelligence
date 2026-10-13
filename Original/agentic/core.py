"""Shared business logic. No patient-care use. Never fabricate missing measurements."""
import json, os, sqlite3, uuid
from datetime import datetime, timezone
from pathlib import Path

FEATURES = ['age','systolic_bp','diastolic_bp','heart_rate','respiratory_rate','temperature_c','spo2','pain_score','gcs_total','arrival_mode','mental_status_triage']
NUMERIC = FEATURES[:9]
BASE = Path(__file__).resolve().parents[1]
MODEL = Path(os.getenv('TRIAGE_MODEL_PATH', str(BASE/'models'/'triagegeist_lightgbm_research.joblib')))
DB = Path(os.getenv('TRIAGE_AUDIT_DB',str(BASE/'demo_audit.sqlite3')))

def now(): return datetime.now(timezone.utc).isoformat()
def connect():
    DB.parent.mkdir(parents=True,exist_ok=True)
    con=sqlite3.connect(DB)
    con.execute('CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, created_at TEXT, status TEXT, assessment_json TEXT, prediction_json TEXT, review_json TEXT)')
    con.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT, at TEXT, actor TEXT, event TEXT, detail_json TEXT)')
    con.commit();return con

def log_event(case_id,actor,event,details):
    with connect() as con:
        con.execute('INSERT INTO events (case_id,at,actor,event,detail_json) VALUES (?,?,?,?,?)',(case_id,now(),actor,event,json.dumps(details,default=str)))

def validate(assessment):
    errors=[]; missing=[]; warnings=[]
    for key in NUMERIC:
        val=assessment.get(key)
        if val is None or val=='': missing.append(key);continue
        try: val=float(val)
        except (TypeError,ValueError): errors.append(f'{key} must be a number');continue
        if not (val == val and abs(val)!=float('inf')): errors.append(f'{key} is not finite')
    for key in FEATURES[9:]:
        if not assessment.get(key): missing.append(key)
    limits={'age':(0,120),'systolic_bp':(30,300),'diastolic_bp':(20,200),'heart_rate':(20,250),'respiratory_rate':(4,80),'temperature_c':(25,45),'spo2':(0,100),'pain_score':(0,10),'gcs_total':(3,15)}
    for k,(lo,hi) in limits.items():
        v=assessment.get(k)
        if v is not None and v!='':
            try:
                if not lo<=float(v)<=hi: errors.append(f'{k} outside demo plausibility range')
            except (ValueError,TypeError): pass
    try:
        if assessment.get('systolic_bp') is not None and assessment.get('diastolic_bp') is not None and float(assessment['diastolic_bp'])>=float(assessment['systolic_bp']):errors.append('Diastolic BP must be below systolic BP')
    except (ValueError,TypeError):pass
    if missing:warnings.append('Incomplete assessment; model inference withheld. Missing data is not assumed normal.')
    return {'ok_to_predict':not errors and not missing,'errors':errors,'missing':missing,'warnings':warnings,'mode':'fictional-research-only'}

from functools import lru_cache

@lru_cache(maxsize=1)
def cached_model():
    import joblib
    return joblib.load(MODEL)

def predict(assessment):
    import pandas as pd
    if not MODEL.is_file():return {'ok':False,'error':f'Model missing at {MODEL}. Copy trained .joblib to models/.'}
    v=validate(assessment)
    if not v['ok_to_predict']:return {'ok':False,'error':'Assessment not complete or valid','quality':v}
    blob=cached_model()
    if not isinstance(blob,dict) or 'pipeline' not in blob or 'features' not in blob or blob.get('research_only') is not True:
        return {'ok':False,'error':'Incompatible or not explicitly research-only model artifact'}
    feats=blob['features']; missing_features=[x for x in feats if x not in assessment]
    if missing_features:return {'ok':False,'error':'Feature schema mismatch','missing_features':missing_features}
    p=blob['pipeline']; data=pd.DataFrame([{k:assessment.get(k) for k in feats}],columns=feats)
    probs=p.predict_proba(data)[0]; cls=int(p.classes_[int(probs.argmax())])
    return {'ok':True,'experimental_class':cls,'synthetic_model_probability':round(float(max(probs)),5),'class_probabilities':{str(c):round(float(x),5) for c,x in zip(p.classes_,probs)},'model_file':MODEL.name,'evidence':'Synthetic Triagegeist examples only; NOT validated on patients','real_world_queue_action':False}

def draft(assessment,prediction,quality,actor='demo-worker'):
    case_id=str(uuid.uuid4())
    report={'case_id':case_id,'quality':quality,'experimental_prediction':prediction,'message':'RESEARCH ONLY. Fictional case. Clinical interpretation and real patient prioritisation are not supported.','pending_review':True}
    with connect() as con:
        con.execute('INSERT INTO cases VALUES (?,?,?,?,?,?)',(case_id,now(),'PENDING_CLINICIAN_REVIEW',json.dumps(assessment),json.dumps(prediction),None))
    log_event(case_id,actor,'case_created',{'status':'PENDING_CLINICIAN_REVIEW'})
    log_event(case_id,'mcp-agent','data_validation',quality)
    log_event(case_id,'mcp-agent','research_prediction',prediction)
    log_event(case_id,'mcp-agent','draft_created',{'status':'PENDING_CLINICIAN_REVIEW'})
    return report

def review(case_id,reviewer,decision,note):
    allowed=('approved_demo_record','needs_more_information','rejected')
    if decision not in allowed:return {'ok':False,'error':f'Decision must be one of {allowed}'}
    if not reviewer.strip():return {'ok':False,'error':'Reviewer identity is required'}
    with connect() as con:
        row=con.execute('SELECT status FROM cases WHERE case_id=?',(case_id,)).fetchone()
        if not row:return {'ok':False,'error':'Unknown case ID'}
        if row[0]!='PENDING_CLINICIAN_REVIEW':return {'ok':False,'error':'Case already reviewed'}
        status={'approved_demo_record':'APPROVED_DEMO_ONLY','needs_more_information':'MORE_INFORMATION_REQUIRED','rejected':'REJECTED'}[decision]
        con.execute('UPDATE cases SET status=?,review_json=? WHERE case_id=?',(status,json.dumps({'reviewer':reviewer,'decision':decision,'note':note,'time':now()}),case_id))
    log_event(case_id,reviewer,'clinician_demo_review',{'decision':decision,'status':status,'note':note})
    return {'ok':True,'case_id':case_id,'status':status,'real_patient_queue_changed':False}

def history(case_id):
    with connect() as con:
        case=con.execute('SELECT case_id,created_at,status,assessment_json,prediction_json,review_json FROM cases WHERE case_id=?',(case_id,)).fetchone()
        events=con.execute('SELECT at,actor,event,detail_json FROM events WHERE case_id=? ORDER BY id',(case_id,)).fetchall()
    if not case:return {'ok':False,'error':'Unknown case'}
    return {'ok':True,'case_id':case[0],'created_at':case[1],'status':case[2],'assessment':json.loads(case[3]),'prediction':json.loads(case[4]),'review':json.loads(case[5]) if case[5] else None,'events':[{'at':e[0],'actor':e[1],'event':e[2],'detail':json.loads(e[3])} for e in events]}
