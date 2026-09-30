from __future__ import annotations

from pathlib import Path
import json

def log_experiments(churn_metrics: dict, forecast_metrics: dict, model_dir: str, artifact_dir: str) -> dict:
    """Log key model metrics to MLflow when installed; otherwise record a clear install requirement."""
    result={"mlflow_available":False,"status":"not_run","message":"Install mlflow to enable experiment tracking."}
    try:
        import mlflow
        mlflow.set_experiment("RetailPulse")
        with mlflow.start_run(run_name="churn") as run:
            mlflow.log_params({"model":churn_metrics.get("model_name",""),"random_state":42})
            for k in ["held_out_roc_auc","precision","recall","f1_score","accuracy","cv_auc_mean","cv_auc_std"]:
                if k in churn_metrics: mlflow.log_metric(k,float(churn_metrics[k]))
            mlflow.log_artifacts(str(artifact_dir),artifact_path="churn_artifacts")
            result={"mlflow_available":True,"status":"logged","churn_run_id":run.info.run_id}
        with mlflow.start_run(run_name="forecast") as run2:
            mlflow.log_params({"model":forecast_metrics.get("model_name",""),"horizon_months":3})
            for k in ["backtest_mae","backtest_rmse","backtest_mape"]:
                if k in forecast_metrics: mlflow.log_metric(k,float(forecast_metrics[k]))
            result["forecast_run_id"]=run2.info.run_id
        return result
    except ImportError:
        return result
    except Exception as exc:
        result.update({"status":"error","message":str(exc)})
        return result
