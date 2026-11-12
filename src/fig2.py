import sys; sys.path.insert(0,'src')
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import pandas as pd
SH=pd.read_csv("results/shap_check.csv")
NM={"SIZE":"SIZE (ln total assets)","DTI_ST_S_DTI":"ST – Digital strategy","DTI_UX_S_DTI":"UX – Customer experience","Year":"Year","DTI_CU_S_DTI":"CU – Digital culture","DTI_PL_S_DTI":"PL – Infrastructure/platform","DTI_OP_S_DTI":"OP – Digital operations","DTI_DA_S_DTI":"DA – Data and information assets","Own":"Ownership (state = 1)"}
sa=SH[SH.target=="ROA"].set_index("feature"); se=SH[SH.target=="ROE"].set_index("feature")
order=sa.in_sample.sort_values(ascending=False).index.tolist()
names=[NM[k] for k in order]
roa_in=sa.loc[order,"in_sample"].tolist(); roa_ho=sa.loc[order,"held_out_oof"].tolist()
roe_in=se.loc[order,"in_sample"].tolist(); roe_ho=se.loc[order,"held_out_oof"].tolist()
from common import ml_sample, load
NFIT=len(ml_sample(load()))
plt.rcParams["font.family"]="DejaVu Sans"
fig,axs=plt.subplots(1,2,figsize=(4.8,2.9),sharey=True)
y=np.arange(len(names))[::-1]
for ax,a,b,t in ((axs[0],roa_in,roa_ho,"ROA"),(axs[1],roe_in,roe_ho,"ROE")):
    ax.barh(y+0.2,a,0.38,color="#2b5c8a",label=f"In-sample (fit on all {NFIT})"); ax.barh(y-0.2,b,0.38,color="#d98c3a",label="Held-out (out-of-fold)")
    ax.set_title(t,fontsize=8,fontweight="bold"); ax.set_xlabel("Mean |SHAP|",fontsize=7); ax.tick_params(labelsize=6.5)
    ax.spines[["top","right"]].set_visible(False)
axs[0].set_yticks(y); axs[0].set_yticklabels(names,fontsize=6.5)
axs[0].legend(fontsize=6,loc="lower right",frameon=False)
fig.tight_layout(); fig.savefig("figures/fig2_shap.png",dpi=600); fig.savefig("figures/fig2_shap.pdf")
