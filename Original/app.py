"""Triage Intelligence — clinical priority assessment interface.
Prediction uses the existing LangGraph/MCP LightGBM integration.
"""
import asyncio
import html
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from agentic.core import validate, MODEL
from agentic.workflow import run

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'demo_register.sqlite3'
LEVELS = {
    1: ('Red', '#BE123C', '#FFF1F2'),
    2: ('Orange', '#C2410C', '#FFF7ED'),
    3: ('Yellow', '#92400E', '#FEF9C3'),
    4: ('Green', '#166534', '#F0FDF4'),
    5: ('Teal', '#0F766E', '#F0FDFA'),
}
FIELDS = {'age':'Age','systolic_bp':'BP (systolic)','diastolic_bp':'BP (diastolic)',
          'heart_rate':'Pulse','respiratory_rate':'Respiratory rate','temperature_c':'Temperature',
          'spo2':'SpO₂','pain_score':'Pain score','gcs_total':'GCS score',
          'arrival_mode':'Arrival method','mental_status_triage':'Consciousness'}
FACILITIES = ['Primary Healthcare Clinic','Community Health Centre','District Hospital',
              'Regional Hospital','Provincial Tertiary Hospital','Central Hospital',
              'Specialised Hospital','Private Hospital / Clinic']


def connect():
    con = sqlite3.connect(DB, timeout=12)
    con.execute('''CREATE TABLE IF NOT EXISTS patient_demo_register (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        day TEXT NOT NULL, facility TEXT NOT NULL, facility_type TEXT NOT NULL,
        patient_number INTEGER NOT NULL, recorded_at TEXT NOT NULL,
        worker TEXT NOT NULL, experimental_level INTEGER, model_probability REAL,
        case_id TEXT NOT NULL UNIQUE, status TEXT NOT NULL DEFAULT 'Recorded',
        UNIQUE(day,facility,patient_number)
    )''')
    con.commit()
    return con


def next_number(facility, day):
    with connect() as con:
        row = con.execute('SELECT COALESCE(MAX(patient_number),0)+1 FROM patient_demo_register WHERE facility=? AND day=?', (facility,day)).fetchone()
    return int(row[0])


def records(facility, day):
    with connect() as con:
        return pd.read_sql_query('''SELECT patient_number AS "Patient No.",
            COALESCE('Level '||experimental_level,'Not available') AS "Clinical priority",
            CASE experimental_level WHEN 1 THEN 'Red' WHEN 2 THEN 'Orange' WHEN 3 THEN 'Yellow'
                 WHEN 4 THEN 'Green' WHEN 5 THEN 'Teal' ELSE '—' END AS "Colour",
            model_probability AS "Model probability", status AS "Status",
            recorded_at AS "Recorded" FROM patient_demo_register
            WHERE facility=? AND day=? ORDER BY patient_number DESC''', con, params=(facility,day))


def save_record(facility, facility_type, worker, result):
    pred = result.get('prediction') or {}
    case_id = (result.get('draft') or {}).get('case_id')
    if not case_id:
        raise ValueError('Agent did not return an auditable case ID; record not saved')
    today = date.today().isoformat()
    with connect() as con:
        n = next_number(facility, today)
        con.execute('''INSERT INTO patient_demo_register
          (day, facility, facility_type, patient_number, recorded_at, worker,
           experimental_level, model_probability, case_id) VALUES (?,?,?,?,?,?,?,?,?)''',
           (today,facility,facility_type,n,datetime.now().strftime('%H:%M:%S'),worker,
            int(pred['experimental_class']) if pred.get('ok') else None,
            float(pred['synthetic_model_probability']) if pred.get('ok') else None,case_id))
    return n


st.set_page_config(page_title='Triage Intelligence', page_icon='🏥', layout='wide')
st.markdown('''<style>
/* Clear typography in the form, sidebar, dropdowns and the report. */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  background-color: #F4F7FB !important;
  color: #163047 !important;
}
[data-testid="stSidebar"], [data-testid="stSidebarContent"] {
  background: #123451 !important;
  color: #FFFFFF !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] small,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
  color: #FFFFFF !important; font-weight: 700 !important;
}
.block-container {max-width: 1140px; padding-top: 1.6rem;}
h1,h2,h3,h4, p, .stMarkdown {color:#163047;}
.ti-title{font-size:2.2rem;font-weight:850;letter-spacing:-.03em;color:#123451;}
.ti-muted{font-size:.95rem;color:#425D72;margin:3px 0 18px;font-weight:600;}
.ti-result{border:1px solid #C8D7E6;border-radius:16px;padding:22px;background:#FFFFFF;margin:15px 0;color:#163047;}
.ti-level{font-size:1.9rem;font-weight:850;padding:13px 20px;border-radius:12px;display:inline-block;}
[data-testid="stForm"]{background:#FFFFFF;border:1px solid #D8E3EF;border-radius:16px;padding:20px;}
/* Labels always contrast with their panel. */
[data-testid="stForm"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stForm"] label, [data-testid="stForm"] p {
 color:#163047 !important; font-weight:750 !important;
}
/* Inputs and selected values: dark, bold text on white. */
[data-baseweb="input"], [data-baseweb="select"],
[data-baseweb="textarea"], [data-baseweb="base-input"],
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"], [data-baseweb="list"] {
 background-color:#FFFFFF !important; color:#102C45 !important;
}
[data-baseweb="input"] input, [data-baseweb="base-input"] input,
[data-baseweb="select"] input,
[data-baseweb="select"] [role="combobox"],
[data-baseweb="select"] [role="combobox"] *,
[data-baseweb="select"] [data-testid="stMarkdownContainer"],
textarea, [data-baseweb="textarea"] textarea,
[data-baseweb="select"] span,
[data-baseweb="tag"] span {
 color:#102C45 !important; font-weight:750 !important;
 -webkit-text-fill-color:#102C45 !important;
}
[data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="base-input"] > div {
 background-color:#FFFFFF !important; color:#102C45 !important;
 border-color:#92A7BA !important;
}
/* Streamlit renders dropdown options in a portal outside the sidebar. */
[role="listbox"], [role="listbox"] *,
[role="option"], [role="option"] *,
[data-baseweb="menu"] *, [data-baseweb="popover"] * {
 color:#102C45 !important; font-weight:700 !important;
 -webkit-text-fill-color:#102C45 !important;
}
[role="option"]:hover, [aria-selected="true"][role="option"] {
 background-color:#DBEAFE !important;
}
[data-testid="stDataFrame"] {background:#FFFFFF;color:#102C45;}
[data-testid="stTabs"] button {font-weight:800 !important;}
</style>''', unsafe_allow_html=True)

st.markdown('<div class="ti-title">🏥 Triage Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="ti-muted">Clinical Priority Assessment</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header('Facility')
    facility_type = st.selectbox('Facility type', FACILITIES)
    facility = st.text_input('Facility name',value='Malamulele').strip()
    worker = st.text_input('Healthcare worker',value='Healthcare worker').strip()

if 'last_result' not in st.session_state:
    st.session_state.last_result = None

if not MODEL.is_file():
    st.error('Trained model not found. Copy triagegeist_lightgbm_research.joblib into the models folder to enable assessment.')
    with st.expander('Model file location'):
        st.code(str(MODEL))

assess_tab, report_tab = st.tabs(['🩺 Patient assessment','📋 Report'])
with assess_tab:
    today = date.today().isoformat()
    if not facility or not worker:
        st.warning('Enter the facility and healthcare worker in the sidebar.')
    else:
        st.subheader(f'Patient {next_number(facility,today):03d}')
        

    with st.form('capture', clear_on_submit=False):
        st.markdown('#### Patient details')
        c1,c2=st.columns(2)
        with c1: age=st.number_input('Age (years)',min_value=0,max_value=120,value=35)
        with c2: arrival=st.selectbox('Arrival method',['walk-in','ambulance','assisted','transfer','other'])
        st.markdown('#### Vital signs')
        a,b=st.columns(2)
        with a:
            st.markdown('**Blood pressure (BP)**')
            bp1,bp2=st.columns(2)
            with bp1: systolic=st.number_input('Systolic',min_value=30,max_value=300,value=120)
            with bp2: diastolic=st.number_input('Diastolic',min_value=20,max_value=200,value=80)
            st.caption(f'BP: {systolic}/{diastolic} mmHg')
            pulse=st.number_input('Pulse (beats/min)',min_value=20,max_value=250,value=80)
            resp=st.number_input('Respiratory rate (/min)',min_value=4,max_value=80,value=18)
        with b:
            temp=st.number_input('Temperature (°C)',min_value=25.0,max_value=45.0,value=37.0,step=.1)
            spo2=st.number_input('Oxygen saturation SpO₂ (%)',min_value=0,max_value=100,value=98)
            pain=st.select_slider('Pain score',options=list(range(11)),value=0)
        st.markdown('#### Observations')
        d,e=st.columns(2)
        with d: mental=st.selectbox('Consciousness',['alert','drowsy','confused','unresponsive'])
        with e:
            gcs_choice=st.selectbox('GCS score (only if actually assessed)', ['Not assessed']+[str(i) for i in range(3,16)])
        obs=st.multiselect('Observed concerns',['Visible injury','Limping','Visible distress','Breathing difficulty','Confusion','Difficulty speaking','Other'])
        note=st.text_input('Brief reason / observation (optional)',placeholder='No names or identifying details')
        submit=st.form_submit_button('Assess & Record',type='primary',use_container_width=True)

    if submit:
        if not facility or not worker:
            st.error('Please enter facility name and healthcare worker in the sidebar.')
        else:
            assessment={'age':age,'systolic_bp':systolic,'diastolic_bp':diastolic,
             'heart_rate':pulse,'respiratory_rate':resp,'temperature_c':temp,
             'spo2':spo2,'pain_score':pain,
             'gcs_total':None if gcs_choice=='Not assessed' else int(gcs_choice),
             'arrival_mode':arrival,'mental_status_triage':mental,
             'observations':obs,'notes':note}
            # Clear human-readable checks before starting the prediction worker.
            # GCS 3–15 is valid. A conflicting consciousness observation is
            # highlighted for verification, but must not suppress the assessment
            # or erase a potentially critical score.
            gcs_conflict = (mental == 'alert' and gcs_choice != 'Not assessed'
                            and int(gcs_choice) < 15)
            if gcs_conflict:
                st.warning('Please verify: consciousness is Alert, but GCS is '
                           + gcs_choice + '. The entered score will be retained. '
                           'Arrange prompt clinical review, especially for low GCS.')
            quality=validate(assessment)
            if not quality['ok_to_predict']:
                st.session_state.last_result=None
                missing=', '.join(FIELDS.get(k,k) for k in quality.get('missing',[]))
                if missing:
                    st.warning('Please complete: '+missing+'.')
                for issue in quality.get('errors',[]): st.error(issue)
            else:
                try:
                    with st.spinner('Assessing patient…'):
                        result=asyncio.run(run(assessment,actor=worker))
                    if not (result.get('prediction') or {}).get('ok'):
                        pred_error = result.get('prediction') or {}
                        st.error('Assessment could not be completed: ' + str(pred_error.get('error') or pred_error.get('reason') or 'Prediction service unavailable.'))
                        if pred_error.get('details'):
                            with st.expander('Error details'):
                                st.code(str(pred_error['details']))
                    else:
                        number=save_record(facility,facility_type,worker,result)
                        st.session_state.last_result={'number':number,'facility':facility,'result':result,'day':today}
                except Exception as exc:
                    st.error('Assessment could not be completed. Nothing was added to the report.')
                    with st.expander('Technical details (developer only)'):st.exception(exc)

    last=st.session_state.last_result
    if last and last['facility']==facility and last['day']==today:
        pred=last['result']['prediction']
        level=int(pred['experimental_class'])
        label,fg,bg=LEVELS[level]
        st.markdown(f'''<div class="ti-result"><div style="font-size:13px;color:#526A80">PATIENT {last['number']:03d} · CLINICAL PRIORITY</div>
         <div style="margin:14px 0"><span class="ti-level" style="color:{fg};background:{bg}">{html.escape(label)} · Level {level}</span></div>
         <div><b>Prediction confidence:</b> {float(pred['synthetic_model_probability']):.1%}</div>
         <div style="color:#526A80;margin-top:9px">Assessment recorded successfully</div></div>''',unsafe_allow_html=True)
        if st.button('Clear result for next patient'):
            st.session_state.last_result=None
            st.rerun()

with report_tab:
    st.subheader('Patient report')
    
    if not facility:
        st.info('Enter a facility name in the sidebar to see its report.')
    else:
        df=records(facility,date.today().isoformat())
        if df.empty:
            st.info('No assessments recorded for this facility today.')
        else:
            formatted=df.copy()
            formatted['Model probability']=formatted['Model probability'].map(lambda x: f'{x:.1%}' if pd.notna(x) else '—')
            st.dataframe(formatted,hide_index=True,use_container_width=True)
            st.download_button('Export report (CSV)',data=df.to_csv(index=False),file_name='triage_patient_report.csv',mime='text/csv')
            st.caption(f'{len(df)} assessment(s) recorded today.')
