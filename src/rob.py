from reg import *
R=[]
MEAS=["S_DTI","LnDTI","W_DTI"]
# (1) length sensitivity
print("corr with ln(words):",{x:round(P[x].corr(P.lnW),3) for x in MEAS+["W_norm"]})
for y in("ROA_w","ROE_w"):
    for x in MEAS: R.append(est(P,y,x,label="A. Length: baseline"))
    R.append(est(P,y,"W_norm",label="A. Length: W_DTI per 1,000 tokens"))
    for x in("LnDTI","W_DTI","S_DTI"): R.append(est(P,y,x,ctrl=("SIZE","lnW"),label="A. Length: + ln(words) control"))
# (2) lags (within bank, consecutive years only)
Ps=P.sort_values(["Bank","Year"]).copy()
for L in (1,2):
    for x in MEAS:
        Ps[f"{x}_L{L}"]=Ps.groupby("Bank")[x].shift(L); ok=Ps.groupby("Bank").Year.shift(L)==Ps.Year-L; Ps.loc[~ok,f"{x}_L{L}"]=np.nan
for L in (1,2):
    for y in("ROA_w","ROE_w"):
        for x in MEAS: R.append(est(Ps,y,f"{x}_L{L}",label=f"B. Lag t-{L}"))
# (3) six channels (LnDTI sub-index, one at a time)
for y in("ROA_w","ROE_w"):
    for c in ["UX","ST","PL","OP","CU","DA"]: R.append(est(P,y,f"DTI_{c}_LnDTI",label="C. Channel (ln sub-index)"))
# (4) ownership + year FE (pooled, clustered)
for y in("ROA_w","ROE_w"):
    for x in MEAS: R.append(est(P,y,x,yearfe=True,own=True,label="D. + Ownership + year dummies (pooled)"))
# (5) contamination checks
Q=P[~((P.Bank=="BID")&(P.Year==2023))]
for y in("ROA_w","ROE_w"):
    for x in MEAS: R.append(est(Q,y,x,label="E. Drop BID 2023 (ROA 18.83)"))
Q2=Q[~((Q.Bank=="BID")&(Q.Year==2025))&~((Q.Bank=="CTG")&(Q.Year==2024))]
for y in("ROA_w","ROE_w"):
    for x in MEAS: R.append(est(Q2,y,x,label="E. + drop duplicate-asset rows (BID 2025, CTG 2024)"))
T=pd.DataFrame(R); T.to_csv("results/robustness_all.csv",index=False)
pd.set_option("display.width",250)
print(T.drop(columns=["se"]).round(3).to_string(index=False))
