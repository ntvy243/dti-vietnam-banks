from common import *
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from scipy import stats
raw=load(); P=panel275(raw)
P["lnW"]=np.log(P.total_words); P["W_norm"]=P.W_DTI/P.total_words*1000   # TF-IDF weight per 1,000 tokens
def haus(d,y,xs):
    X=d[xs]; fe=PanelOLS(d[y],X,entity_effects=True).fit(); re=RandomEffects(d[y],X.assign(c=1.0)).fit()
    b=fe.params-re.params[fe.params.index]; V=fe.cov-re.cov.loc[b.index,b.index]; h=float(b@np.linalg.pinv(V)@b)
    return h,(stats.chi2.sf(h,len(b)) if h>0 else np.nan)
def est(d,y,x,ctrl=("SIZE",),std=True,force=None,yearfe=False,own=False,label=""):
    d=d.dropna(subset=[x,y]).copy()
    if std: d[x]=(d[x]-d[x].mean())/d[x].std()
    d=d.set_index(["Bank","Year"]); xs=[x]+list(ctrl)
    if yearfe or own:
        X=d[xs].copy()
        if own: X["Own"]=d["Own"]
        if yearfe:
            for yv in sorted(d.index.get_level_values(1).unique())[1:]: X[f"y{yv}"]=(d.index.get_level_values(1)==yv).astype(float)
        X["const"]=1.0; m=PooledOLS(d[y],X).fit(cov_type="clustered",cluster_entity=True); name="Pooled"
        hp=np.nan; r2=m.rsquared
    else:
        h,hp=haus(d,y,xs); use="FE" if (force=="FE" or (force is None and hp<0.05)) else "RE"
        if use=="FE": m=PanelOLS(d[y],d[xs],entity_effects=True).fit(cov_type="clustered",cluster_entity=True); r2=m.rsquared; name="FEM"
        else: m=RandomEffects(d[y],d[xs].assign(const=1.0)).fit(cov_type="clustered",cluster_entity=True); r2=m.rsquared; name="REM"
    ci=m.conf_int().loc[x]
    r=dict(spec=label,y=y,x=x,est=name,coef=m.params[x],se=m.std_errors[x],lo=ci.iloc[0],hi=ci.iloc[1],p=m.pvalues[x],R2=r2,N=int(m.nobs),hausman_p=hp)
    if own: r["own_coef"]=m.params["Own"]; r["own_p"]=m.pvalues["Own"]
    return r
if __name__=="__main__":
    rows=[]
    # Table 3 reproduction (winsorised outcomes, standardised DTI, SIZE)
    for y in("ROA","ROE"):
        for x in("S_DTI","LnDTI","W_DTI"): rows.append(est(P,y+"_w",x,label="Main (Table 3)"))
    T=pd.DataFrame(rows); print(T.round(3).to_string(index=False)); T.to_csv("results/main_repro.csv",index=False)
