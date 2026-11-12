from common import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
import shap, json
raw=load(); d=ml_sample(raw).reset_index(drop=True); d["SIZE"]=np.log(d.TotalAssets)
FULL=CH+["SIZE","Own","Year"]; CTRL=["SIZE","Own","Year"]; SZ=["SIZE"]
def mk(k): return make_pipeline(StandardScaler(),LinearRegression()) if k=="lin" else RandomForestRegressor(300,max_depth=5,min_samples_leaf=3,random_state=42)
def sp(scheme):
    if scheme=="shuffled": return list(KFold(5,shuffle=True,random_state=42).split(d))
    if scheme=="grouped": return list(GroupKFold(5).split(d,groups=d.Bank))
    yrs=sorted(d.Year.unique())[-3:]; return [(np.where(d.Year<y)[0],np.where(d.Year==y)[0]) for y in yrs]
def oof(kind,cols,y,folds):
    pr=np.full(len(d),np.nan); fr=[]
    for tr,te in folds:
        m=mk(kind).fit(d.iloc[tr][cols],d.iloc[tr][y]); pr[te]=m.predict(d.iloc[te][cols]); fr.append(r2_score(d.iloc[te][y],pr[te]))
    return pr,np.array(fr)
banks=d.Bank.unique(); rng=np.random.default_rng(1)
idx_by_bank={b:np.where(d.Bank==b)[0] for b in banks}
def boot_diff(y,pa,pb,mask,B=2000):
    out=[]
    for _ in range(B):
        bs=rng.choice(banks,len(banks)); ii=np.concatenate([idx_by_bank[b] for b in bs]); ii=ii[mask[ii]]
        out.append(r2_score(d[y].values[ii],pa[ii])-r2_score(d[y].values[ii],pb[ii]))
    return np.percentile(out,[2.5,97.5])
rows=[]
for y in ["ROA","ROE"]:
    for sch in ["shuffled","grouped","forward"]:
        f=sp(sch); res={k:oof(*k_,y,f) for k,k_ in {"lin":("lin",FULL),"rf_full":("rf",FULL),"rf_ctrl":("rf",CTRL),"rf_size":("rf",SZ)}.items()}
        mask=~np.isnan(res["rf_full"][0])
        for name,(a,b) in {"RF_full - controls":("rf_full","rf_ctrl"),"controls - SIZE only":("rf_ctrl","rf_size"),"RF_full - SIZE only":("rf_full","rf_size"),"RF_full - linear":("rf_full","lin")}.items():
            pa,pb=res[a][0],res[b][0]
            point=r2_score(d[y][mask],pa[mask])-r2_score(d[y][mask],pb[mask])
            lo,hi=boot_diff(y,pa,pb,mask)
            fd=res[a][1]-res[b][1]
            rows.append(dict(target=y,design=sch,contrast=name,delta_pooled=round(point,3),ci_lo=round(lo,3),ci_hi=round(hi,3),fold_mean=round(fd.mean(),3),fold_sd=round(fd.std(),3),n_folds=len(fd)))
        if sch=="shuffled":
            for k in res: rows.append(dict(target=y,design="shuffled",contrast="R2 fold mean±sd: "+k,delta_pooled=round(res[k][1].mean(),3),ci_lo=round(res[k][1].std(),3),ci_hi=None,fold_mean=None,fold_sd=None,n_folds=5))
t=pd.DataFrame(rows); t.to_csv("results/ablation_ci.csv",index=False); print(t.to_string(index=False))
# ---- SHAP: in-sample (fit on all) vs held-out (out-of-fold)
shp={}
for y in ["ROA","ROE"]:
    m=mk("rf").fit(d[FULL],d[y]); sv=shap.TreeExplainer(m).shap_values(d[FULL]); ins=np.abs(sv).mean(0)
    oo=np.zeros(len(FULL)); 
    S=np.zeros((len(d),len(FULL)))
    for tr,te in sp("shuffled"):
        mm=mk("rf").fit(d.iloc[tr][FULL],d.iloc[tr][y]); S[te]=shap.TreeExplainer(mm).shap_values(d.iloc[te][FULL])
    shp[y]=pd.DataFrame({"feature":FULL,"in_sample":ins.round(3),"held_out_oof":np.abs(S).mean(0).round(3)})
    print(y);print(shp[y].to_string(index=False))
pd.concat(shp,names=["target"]).to_csv("results/shap_check.csv")
