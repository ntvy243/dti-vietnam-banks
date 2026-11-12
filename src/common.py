import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
RAW="data/panel_dti_financials.csv"
CH=["DTI_UX_S_DTI","DTI_ST_S_DTI","DTI_PL_S_DTI","DTI_OP_S_DTI","DTI_CU_S_DTI","DTI_DA_S_DTI"]
STATE=['BID','CTG','VCB']
def load():
    raw=pd.read_csv(RAW); raw=raw[raw.ocr_no_diacritics==0].copy()   # 17 OCR files without usable diacritics are excluded (measurement failure, not zero disclosure)
    raw["Own"]=raw.Bank.isin(STATE).astype(int)
    raw["unit_text"]=raw.TotalAssets>10000
    return raw
def ml_sample(raw): return raw[~raw.unit_text].copy()           # as analysed in the paper (N=231 after the OCR screen)
def panel275(raw):
    d=raw.copy(); d.loc[d.unit_text,"TotalAssets"]/=1000
    d=d[~((d.Bank=="BID")&d.Year.isin([2016,2019,2022]))].copy()
    d["SIZE"]=np.log(d.TotalAssets)
    for v in("ROA","ROE"): d[v+"_w"]=d[v].clip(d[v].quantile(.01),d[v].quantile(.99))
    return d
