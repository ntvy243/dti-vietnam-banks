import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
raw=pd.read_csv("data/panel_dti_financials.csv"); raw=raw[raw.ocr_no_diacritics==0].copy()   # 17 OCR files without usable diacritics are excluded
CH=["DTI_UX_S_DTI","DTI_ST_S_DTI","DTI_PL_S_DTI","DTI_OP_S_DTI","DTI_CU_S_DTI","DTI_DA_S_DTI"]
STATE=['BID','CTG','VCB']
raw["Own"]=raw.Bank.isin(STATE).astype(int)
# ---------- A. audit of exclusions ----------
raw["unit_mismatch"]=raw.TotalAssets>10000
ex=raw[raw.unit_mismatch]
print("A. rows with TotalAssets in a different unit (excluded from ML):",len(ex))
print(ex.groupby("Bank").size().to_dict())
print("   by ownership: excluded state",int(ex.Own.sum()),"| excluded private",int((1-ex.Own).sum()))
ret=raw[~raw.unit_mismatch]
print("   retained: N",len(ret),"state",int(ret.Own.sum()))
print("   years of retained state rows:",ret[ret.Own==1][["Bank","Year","TotalAssets"]].values.tolist())
print("   share of state-owned bank-years retained: %d/%d"%(ret.Own.sum(),raw.Own.sum()))
print("   mean year retained %.1f excluded %.1f"%(ret.Year.mean(),ex.Year.mean()))
tbl=pd.DataFrame({"retained":ret[["ROA","ROE","S_DTI","LnDTI","Year"]].mean(),"excluded":ex[["ROA","ROE","S_DTI","LnDTI","Year"]].mean()}).round(3); print(tbl)
raw["audit_reason"]=np.where(raw.unit_mismatch,"TotalAssets in different unit (billion vs trillion VND)","")
raw.loc[raw.audit_reason!="","Bank Year TotalAssets ROA ROE Own audit_reason".split()].to_csv("results/ml_exclusions_by_reason.csv",index=False)
# suspicious rows in the 248 retained sample
sus=ret[(ret.Bank=="BID")&ret.Year.isin([2016,2019,2022])][["Bank","Year","TotalAssets"]]
print("   suspicious retained rows (BID assets ~ other banks' values):\n",sus.to_string(index=False))
dups=raw.sort_values(["Bank","Year"]); dups=dups[dups.groupby("Bank").TotalAssets.diff()==0][["Bank","Year","TotalAssets"]]
print("   identical assets to prior year:\n",dups.to_string(index=False))
print("   ROA>10:",raw[raw.ROA>10][["Bank","Year","ROA"]].values.tolist())
# ---------- B. zeros / losses ----------
print("\nB. ROA<0:",int((raw.ROA<0).sum()),"ROA==0:",int((raw.ROA==0).sum()),"| ROE<0:",int((raw.ROE<0).sum()),"ROE==0:",int((raw.ROE==0).sum()))
print(raw[(raw.ROA<=0)|(raw.ROE<=0)][["Bank","Year","ROA","ROE"]].to_string(index=False))
print("p1/p99 ROA",raw.ROA.quantile([.01,.99]).round(3).tolist(),"ROE",raw.ROE.quantile([.01,.99]).round(3).tolist())
# ---------- C/D. validation designs ----------
def mk(kind):
    return make_pipeline(StandardScaler(),LinearRegression()) if kind=="lin" else RandomForestRegressor(300,max_depth=5,min_samples_leaf=3,random_state=42)
def folds(d,scheme):
    if scheme=="shuffled": return list(KFold(5,shuffle=True,random_state=42).split(d))
    if scheme=="bank-grouped": return list(GroupKFold(5).split(d,groups=d.Bank))
    yrs=sorted(d.Year.unique())[-3:]; return [(np.where(d.Year<y)[0],np.where(d.Year==y)[0]) for y in yrs]
def score(kind,cols,y,d,sp,pooled=False):
    sc=[];pr=np.full(len(d),np.nan)
    for tr,te in sp:
        m=mk(kind).fit(d.iloc[tr][cols],d.iloc[tr][y]); p=m.predict(d.iloc[te][cols]); pr[te]=p; sc.append(r2_score(d.iloc[te][y],p))
    ok=~np.isnan(pr); return (r2_score(d[y][ok],pr[ok]) if pooled else np.mean(sc))
def table(d,name):
    d=d.reset_index(drop=True); d["SIZE"]=np.log(d.TotalAssets); rows=[]
    FULL=CH+["SIZE","Own","Year"]; CTRL=["SIZE","Own","Year"]
    for y in ["ROA","ROE"]:
        for sch in ["shuffled","bank-grouped","forward"]:
            sp=folds(d,sch); pooled=(sch=="forward")
            r=dict(sample=name,N=len(d),target=y,design=sch,
                   linear=score("lin",FULL,y,d,sp,pooled),rf_ctrl=score("rf",CTRL,y,d,sp,pooled),rf_full=score("rf",FULL,y,d,sp,pooled))
            r["delta_dti"]=r["rf_full"]-r["rf_ctrl"]; rows.append(r)
    return pd.DataFrame(rows)
paper=raw[~raw.unit_mismatch].copy()
fx=raw.copy(); fx.loc[fx.unit_mismatch,"TotalAssets"]/=1000
fx=fx[~((fx.Bank=="BID")&fx.Year.isin([2016,2019,2022]))&(fx.ROA<10)]   # drop BID wrong-bank asset copies and ROA=18.83 date-serial-type outlier
p245=paper[~((paper.Bank=="BID")&paper.Year.isin([2016,2019,2022]))].copy()   # sensitivity: drop the three suspect BID rows
t=pd.concat([table(paper,"as analysed"),table(fx,"unit-corrected"),table(p245,"drop 3 suspect BID rows")]).round(3)
print("\nC/D.\n",t.to_string(index=False)); t.to_csv("results/table7_validation_designs.csv",index=False)
print("unit-corrected N",len(fx),"state",int(fx.Own.sum()))
# forward test years incl. per-year
