"""Train a research-only LightGBM model on local Triagegeist synthetic cases.
Run: python train_final.py --csv "..\\Compitition\\train.csv"
"""
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier

NUM=['age','systolic_bp','diastolic_bp','heart_rate','respiratory_rate','temperature_c','spo2','pain_score','gcs_total']
CAT=['arrival_mode','mental_status_triage']
FEATURES=NUM+CAT

def metrics(y,p):
    y=np.asarray(y).astype(int); p=np.asarray(p).astype(int)
    urgent=y<=2; predicted=p<=2
    tp=int((urgent&predicted).sum()); fn=int((urgent&~predicted).sum()); fp=int((~urgent&predicted).sum())
    return dict(macro_f1=round(float(f1_score(y,p,average='macro',zero_division=0)),5),balanced_accuracy=round(float(balanced_accuracy_score(y,p)),5),urgent_recall=round(tp/max(1,tp+fn),5),urgent_precision=round(tp/max(1,tp+fp),5),missed_urgent=fn,false_urgent_flags=fp,urgent_predicted_4_or_5=int((urgent&(p>=4)).sum()),confusion_matrix=confusion_matrix(y,p,labels=[1,2,3,4,5]).tolist())

def build():
    pre=ColumnTransformer([
        ('num',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),NUM),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),CAT)
    ])
    return Pipeline([('preprocessor',pre),('classifier',LGBMClassifier(n_estimators=250,learning_rate=.05,num_leaves=20,max_depth=6,reg_lambda=3,random_state=42,verbosity=-1,n_jobs=4))])

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); args=ap.parse_args()
    d=pd.read_csv(args.csv)
    needed=FEATURES+['triage_acuity','site_id']; missing=set(needed)-set(d.columns)
    if missing: raise ValueError(f'Missing required columns: {sorted(missing)}')
    for c in NUM: d[c]=pd.to_numeric(d[c],errors='coerce')
    for c in CAT: d[c]=d[c].fillna('Unknown').astype(str)
    d['triage_acuity']=pd.to_numeric(d['triage_acuity'],errors='coerce')
    d=d[d['triage_acuity'].isin([1,2,3,4,5])].reset_index(drop=True)
    # Fit one deployment model on all labelled synthetic examples AFTER independent holdout evaluation.
    ix=np.arange(len(d)); tr,te=train_test_split(ix,test_size=.2,random_state=42,stratify=d.triage_acuity)
    candidate=build(); candidate.fit(d.iloc[tr][FEATURES],d.iloc[tr].triage_acuity.astype(int))
    result={'random_holdout':metrics(d.iloc[te].triage_acuity,candidate.predict(d.iloc[te][FEATURES]))}
    if d.site_id.nunique()>1:
        atr,ate=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(d,groups=d.site_id))
        site_model=build(); site_model.fit(d.iloc[atr][FEATURES],d.iloc[atr].triage_acuity.astype(int))
        result['held_out_synthetic_sites']=metrics(d.iloc[ate].triage_acuity,site_model.predict(d.iloc[ate][FEATURES]))
    print(json.dumps(result,indent=2))
    final=build(); final.fit(d[FEATURES],d.triage_acuity.astype(int))
    out=Path('models');out.mkdir(exist_ok=True)
    joblib.dump({'pipeline':final,'features':FEATURES,'dataset':'Triagegeist synthetic cases','research_only':True},out/'triagegeist_lightgbm_research.joblib')
    Path('reports').mkdir(exist_ok=True)
    (Path('reports')/'triagegeist_final_evaluation.json').write_text(json.dumps({'source':'Triagegeist synthetic; no patient-level clinical validation','features':FEATURES,'results':result},indent=2),encoding='utf-8')
    print('Saved models/triagegeist_lightgbm_research.joblib')
    print('Research demonstration with fictional patient cases only. Do not use for real patient care.')
if __name__=='__main__':main()
