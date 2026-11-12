"""Table 4: linear, Random Forest and XGBoost on the ML sample (231 observations), identical inputs and folds.
Hyperparameters are fixed (no search). Run from the repository root: python src/table4_models.py"""
import sys; sys.path.insert(0,"src")
from common import *
import xgboost, sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
d=ml_sample(load()).reset_index(drop=True); d["SIZE"]=np.log(d.TotalAssets)
X=CH+["SIZE","Own","Year"]
M={"Linear":lambda:make_pipeline(StandardScaler(),LinearRegression()),
   "Random Forest":lambda:RandomForestRegressor(300,max_depth=5,min_samples_leaf=3,random_state=42),
   "XGBoost":lambda:xgboost.XGBRegressor(n_estimators=300,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,random_state=42)}
rows=[]
for y in ("ROA","ROE"):
    for name,mk in M.items():
        r=[];rm=[];ma=[]
        for tr,te in KFold(5,shuffle=True,random_state=42).split(d):
            p=mk().fit(d.iloc[tr][X],d.iloc[tr][y]).predict(d.iloc[te][X])
            r.append(r2_score(d.iloc[te][y],p)); rm.append(mean_squared_error(d.iloc[te][y],p)**.5); ma.append(mean_absolute_error(d.iloc[te][y],p))
        rows.append(dict(target=y,model=name,R2=round(np.mean(r),3),R2_sd=round(np.std(r),3),RMSE=round(np.mean(rm),3),MAE=round(np.mean(ma),3)))
T=pd.DataFrame(rows); print(T.to_string(index=False)); T.to_csv("results/table4_models.csv",index=False)
print("xgboost",xgboost.__version__,"scikit-learn",sklearn.__version__)
