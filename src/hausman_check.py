import pandas as pd,numpy as np,itertools,warnings;warnings.filterwarnings("ignore")
from linearmodels.panel import PanelOLS,RandomEffects
from scipy import stats
raw=pd.read_csv("data/panel_dti_financials.csv")
raw.loc[raw.TotalAssets>10000,"TotalAssets"]/=1000
def haus(d,y,x):
    d=d.set_index(["Bank","Year"]); X=d[[x,"SIZE"]].copy(); X[x]=(X[x]-X[x].mean())/X[x].std()
    fe=PanelOLS(d[y],X,entity_effects=True).fit(cov_type="clustered",cluster_entity=True)
    fe0=PanelOLS(d[y],X,entity_effects=True).fit()
    re=RandomEffects(d[y],X.assign(c=1.0)).fit()
    b=fe0.params-re.params[fe0.params.index]; V=fe0.cov-re.cov.loc[b.index,b.index]
    h=float(b@np.linalg.pinv(V)@b); return stats.chi2.sf(h,len(b))
def prep(d,wins=True):
    d=d.copy(); d["SIZE"]=np.log(d.TotalAssets)
    if wins:
        for v in("ROA","ROE"): d[v]=d[v].clip(d[v].quantile(.01),d[v].quantile(.99))
    return d
target={("ROA","LnDTI"):.017,("ROA","W_DTI"):.001,("ROE","S_DTI"):.198,("ROE","LnDTI"):.438,("ROE","W_DTI"):.367}
cands={
 "drop BID16/19/22":lambda d:d[~((d.Bank=="BID")&d.Year.isin([2016,2019,2022]))],
 "drop BID16/19/22 first":None}
def test(name,d,wins=True):
    d=prep(d,wins); out={k:round(haus(d,*k),3) for k in target}; out[("ROA","S_DTI")]=round(haus(d,"ROA","S_DTI"),3)
    ok=all(abs(out[k]-v)<.0015 for k,v in target.items())
    print(name,len(d),ok,out)
test("all278",raw)
test("drop BID16/19/22",raw[~((raw.Bank=="BID")&raw.Year.isin([2016,2019,2022]))])
test("drop BID16/19/22 + BID23",raw[~((raw.Bank=="BID")&raw.Year.isin([2016,2019,2022,2023]))])
test("drop BID16/19/22 nowins",raw[~((raw.Bank=="BID")&raw.Year.isin([2016,2019,2022]))],False)
test("drop BID16/19/23",raw[~((raw.Bank=="BID")&raw.Year.isin([2016,2019,2023]))])
test("drop BID16/22/23",raw[~((raw.Bank=="BID")&raw.Year.isin([2016,2022,2023]))])
test("drop BID19/22/23",raw[~((raw.Bank=="BID")&raw.Year.isin([2019,2022,2023]))])
print("----")
d=prep(raw[~((raw.Bank=="BID")&raw.Year.isin([2016,2019,2022]))]).set_index(["Bank","Year"])
X=d[["S_DTI","SIZE"]].copy(); X["S_DTI"]=(X.S_DTI-X.S_DTI.mean())/X.S_DTI.std()
fe=PanelOLS(d.ROA,X,entity_effects=True).fit(); re=RandomEffects(d.ROA,X.assign(c=1.0)).fit()
b=fe.params-re.params[fe.params.index]; V=fe.cov-re.cov.loc[b.index,b.index]
print(b.values, np.linalg.eigvalsh(V.values), float(b@np.linalg.pinv(V)@b))
V2=V.values; print("naive h", float(b.values@np.linalg.inv(V2)@b.values))
