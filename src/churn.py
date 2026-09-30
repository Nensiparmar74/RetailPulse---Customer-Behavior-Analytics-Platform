from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

FEATURES=["recency","frequency","monetary","aov","total_quantity","unique_products","lifetime_days","avg_days_between","return_ratio"]

def build_churn_dataset(sales: pd.DataFrame, deduped_raw: pd.DataFrame, obs_end: str="2011-06-30 23:59:59", pred_end: str="2011-09-30 23:59:59") -> tuple[pd.DataFrame,dict]:
    """Create a leakage-free customer churn dataset using an observation and future prediction window."""
    obs_cut=pd.Timestamp(obs_end); pred_cut=pd.Timestamp(pred_end)
    obs=sales[sales["InvoiceDate"]<=obs_cut].copy(); future=sales[(sales["InvoiceDate"]>obs_cut)&(sales["InvoiceDate"]<=pred_cut)].copy()
    if obs.empty or future.empty: raise ValueError("Churn observation or prediction window is empty.")
    g=obs.groupby("Customer ID")
    c=g.agg(last_purchase=("InvoiceDate","max"),first_purchase=("InvoiceDate","min"),frequency=("Invoice","nunique"),monetary=("Revenue","sum"),total_quantity=("Quantity","sum"),unique_products=("StockCode","nunique")).reset_index()
    c["recency"]=(obs_cut-c["last_purchase"]).dt.days.astype(int)
    c["lifetime_days"]=(c["last_purchase"]-c["first_purchase"]).dt.days.clip(lower=1)
    c["aov"]=c["monetary"]/c["frequency"].replace(0,np.nan)
    c["avg_days_between"]=np.where(c["frequency"]>1,c["lifetime_days"]/(c["frequency"]-1),c["lifetime_days"])
    r=deduped_raw[(deduped_raw["InvoiceDate"]<=obs_cut)&deduped_raw["Customer ID"].notna()].groupby("Customer ID").agg(return_qty=("Quantity",lambda s:float(s[s<0].abs().sum())),positive_qty=("Quantity",lambda s:float(s[s>0].sum())))
    c=c.merge(r,left_on="Customer ID",right_index=True,how="left")
    c["return_ratio"]=(c["return_qty"]/c["positive_qty"].replace(0,np.nan)).fillna(0)
    future_orders=future.groupby("Customer ID")["Invoice"].nunique()
    c["future_orders"]=c["Customer ID"].map(future_orders).fillna(0).astype(int)
    c["churn"]=(c["future_orders"]==0).astype(int)
    meta={"observation_end":str(obs_cut),"prediction_end":str(pred_cut),"churn_definition":"Customer is labelled churned when they have zero completed purchase orders during the three-month prediction window after the observation cutoff.","observation_customers":int(len(c)),"churn_rate":float(c["churn"].mean()*100),"positive_sales_observation_rows":int(len(obs)),"future_window_rows":int(len(future))}
    return c.replace([np.inf,-np.inf],np.nan),meta

def train_churn_model(df: pd.DataFrame, model_dir: str | Path, shap_dir: str | Path) -> tuple[dict,RandomForestClassifier,pd.DataFrame]:
    """Train/evaluate a Random Forest churn classifier and generate SHAP artifacts."""
    model_dir=Path(model_dir); shap_dir=Path(shap_dir); model_dir.mkdir(parents=True,exist_ok=True); shap_dir.mkdir(parents=True,exist_ok=True)
    X=df[FEATURES].fillna(0).astype(float); y=df["churn"].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    model=RandomForestClassifier(n_estimators=250,max_depth=8,min_samples_leaf=3,class_weight="balanced_subsample",random_state=42,n_jobs=-1)
    cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=42)
    cv_scores=cross_val_score(model,Xtr,ytr,cv=cv,scoring="roc_auc",n_jobs=-1)
    model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]; pred=(prob>=0.5).astype(int)
    auc=roc_auc_score(yte,prob); precision=precision_score(yte,pred,zero_division=0); recall=recall_score(yte,pred,zero_division=0); f1=f1_score(yte,pred,zero_division=0); acc=accuracy_score(yte,pred)
    tn,fp,fn,tp=confusion_matrix(yte,pred).ravel(); fpr,tpr,_=roc_curve(yte,prob)
    figures_dir=Path(shap_dir).parent / "figures" / "churn"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5)); plt.plot(fpr,tpr,label=f"ROC-AUC={auc:.3f}"); plt.plot([0,1],[0,1],linestyle="--",label="Chance"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("Churn ROC Curve"); plt.legend(); plt.figtext(0.99,0.01,"Source: RetailPulse held-out churn test set",ha="right",fontsize=7,color="gray"); plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig(figures_dir/"roc_curve.png",dpi=160,bbox_inches="tight"); plt.close()
    cm=np.array([[tn,fp],[fn,tp]])
    plt.figure(figsize=(5,4)); plt.imshow(cm,interpolation="nearest",cmap="Blues"); plt.colorbar(); plt.xticks([0,1],["Not Churned","Churned"],rotation=25); plt.yticks([0,1],["Not Churned","Churned"]); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Churn Confusion Matrix");
    for i in range(2):
        for j in range(2): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.figtext(0.99,0.01,"Source: RetailPulse held-out churn test set",ha="right",fontsize=7,color="gray"); plt.tight_layout(rect=[0,0.03,1,1]); plt.savefig(figures_dir/"confusion_matrix.png",dpi=160,bbox_inches="tight"); plt.close()
    # SHAP compatibility across versions returning either list or 3D array.
    Xshap=Xte.sample(min(400,len(Xte)), random_state=42)
    explainer=shap.TreeExplainer(model); raw_sv=explainer.shap_values(Xshap)
    arr=np.asarray(raw_sv)
    if arr.ndim==3: sv=arr[:,:,1]
    elif isinstance(raw_sv,list): sv=np.asarray(raw_sv[1])
    else: sv=arr
    mean_abs=np.abs(sv).mean(axis=0)
    order=np.argsort(-mean_abs)
    plt.figure(figsize=(8,5)); shap.summary_plot(sv,Xshap,show=False); plt.title("SHAP Beeswarm - Churn Model"); plt.tight_layout(); plt.savefig(shap_dir/"shap_beeswarm.png",dpi=160,bbox_inches="tight"); plt.close()
    plt.figure(figsize=(8,5)); shap.summary_plot(sv,Xshap,plot_type="bar",show=False); plt.title("SHAP Feature Importance - Churn Model"); plt.tight_layout(); plt.savefig(shap_dir/"shap_bar.png",dpi=160,bbox_inches="tight"); plt.close()
    # Representative high-risk observation waterfall
    high_idx=int(np.argmax(model.predict_proba(Xshap)[:,1])); ex=shap.Explanation(values=sv[high_idx],base_values=float(explainer.expected_value[1] if np.asarray(explainer.expected_value).ndim>0 else explainer.expected_value),data=Xshap.iloc[high_idx].to_numpy(),feature_names=FEATURES)
    shap.plots.waterfall(ex,max_display=9,show=False); plt.tight_layout(); plt.savefig(shap_dir/"shap_waterfall_high_risk.png",dpi=160,bbox_inches="tight"); plt.close()
    joblib.dump(model,model_dir/"churn_model.pkl")
    pred_table=df.loc[Xte.index].copy(); pred_table["churn_probability"]=prob; pred_table["predicted_churn"] = pred; pred_table["risk_category"]=pd.cut(pred_table["churn_probability"],bins=[-np.inf,0.33,0.66,np.inf],labels=["Low Risk","Medium Risk","High Risk"])
    metrics={
        "model_name":"Random Forest Churn Classifier", "algorithm":"RandomForestClassifier",
        "held_out_roc_auc":float(auc), "precision":float(precision), "recall":float(recall), "f1_score":float(f1), "accuracy":float(acc),
        "churn_rate":float(y.mean()*100), "test_set_size":int(len(yte)), "cv_auc_mean":float(cv_scores.mean()), "cv_auc_std":float(cv_scores.std()),
        "target_threshold_achieved":bool(auc>0.80),
        "confusion_matrix":{"trueNegative":int(tn),"falsePositive":int(fp),"falseNegative":int(fn),"truePositive":int(tp)},
        "roc_curve":[{"fpr":float(a),"tpr":float(b)} for a,b in zip(fpr,tpr)],
        "feature_importance":[{"feature":f,"importance":float(model.feature_importances_[i]),"description":"Predictive feature importance from the fitted Random Forest."} for i,f in enumerate(FEATURES)],
        "shap_summary":[{"feature":FEATURES[i],"mean_absolute_shap":float(mean_abs[i]),"correlation_with_churn":"positive" if float(np.mean(sv[:,i]))>0 else "negative"} for i in order],
    }
    return metrics,model,pred_table
