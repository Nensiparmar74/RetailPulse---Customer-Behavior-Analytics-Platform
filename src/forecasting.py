from __future__ import annotations

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def monthly_revenue(sales: pd.DataFrame) -> pd.Series:
    """Aggregate positive commercial revenue to month-start frequency."""
    s=sales.set_index("InvoiceDate")["Revenue"].resample("MS").sum().asfreq("MS",fill_value=0)
    return s


def choose_sarima_model(series: pd.Series, validation_months: int=3) -> tuple[dict,pd.DataFrame]:
    """Select a SARIMA configuration using a three-month historical backtest."""
    if len(series)<15: raise ValueError("At least 15 monthly observations are recommended for SARIMA selection.")
    train=series.iloc[:-validation_months]; test=series.iloc[-validation_months:]
    orders=[(0,1,1),(0,0,1),(1,0,0),(1,0,1),(1,1,1),(2,0,0)]
    seasonals=[(1,0,0,12),(0,1,1,12),(0,1,0,12),(1,1,0,12)]
    rows=[]
    for order in orders:
        for seasonal in seasonals:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model=SARIMAX(train,order=order,seasonal_order=seasonal,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
                pred=model.get_forecast(validation_months).predicted_mean
                err=test.to_numpy()-pred.to_numpy()
                mae=float(np.mean(np.abs(err))); rmse=float(np.sqrt(np.mean(err**2))); mape=float(np.mean(np.abs(err)/np.maximum(np.abs(test.to_numpy()),1))*100)
                rows.append({"order":order,"seasonal_order":seasonal,"mae":mae,"rmse":rmse,"mape":mape})
            except Exception:
                continue
    results=pd.DataFrame(rows).sort_values("mape").reset_index(drop=True)
    if results.empty: raise RuntimeError("No SARIMA candidate converged.")
    return results.iloc[0].to_dict(), results


def fit_final_forecast(series: pd.Series, validation_months: int=3) -> tuple[dict,pd.DataFrame,object,pd.DataFrame]:
    """Backtest the selected SARIMA, fit on complete history, and forecast three months."""
    # The workbook ends on 2011-12-09, so December 2011 is treated as incomplete and excluded from final history.
    last_date=pd.Timestamp(series.index.max())
    complete=series.copy()
    if last_date.month==12 and last_date.day < 20:
        complete=complete.iloc[:-1]
    best,_=choose_sarima_model(complete,validation_months=validation_months)
    order=tuple(best["order"]); seasonal=tuple(best["seasonal_order"])
    train=complete.iloc[:-validation_months]; test=complete.iloc[-validation_months:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        back=SARIMAX(train,order=order,seasonal_order=seasonal,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    pred=back.get_forecast(validation_months).predicted_mean; err=test.to_numpy()-pred.to_numpy()
    mae=float(np.mean(np.abs(err))); rmse=float(np.sqrt(np.mean(err**2))); mape=float(np.mean(np.abs(err)/np.maximum(np.abs(test.to_numpy()),1))*100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final=SARIMAX(complete,order=order,seasonal_order=seasonal,enforce_stationarity=False,enforce_invertibility=False).fit(disp=False)
    fc=final.get_forecast(3); ci=fc.conf_int(alpha=.05)
    rows=[{"month":idx.strftime("%Y-%m"),"actual_revenue":float(v),"forecast_revenue":None,"lower_confidence_bound":None,"upper_confidence_bound":None,"is_forecast":False} for idx,v in complete.items()]
    for idx,v,l,u in zip(fc.predicted_mean.index,fc.predicted_mean,ci.iloc[:,0],ci.iloc[:,1]):
        rows.append({"month":idx.strftime("%Y-%m"),"actual_revenue":None,"forecast_revenue":float(v),"lower_confidence_bound":float(l),"upper_confidence_bound":float(u),"is_forecast":True})
    metrics={"model_name":f"SARIMA{order}x{seasonal}","order":order,"seasonal_order":seasonal,"horizon_months":3,"confidence_level":0.95,"backtest_mae":mae,"backtest_rmse":rmse,"backtest_mape":mape,"training_start":complete.index.min().strftime("%Y-%m"),"training_end":complete.index.max().strftime("%Y-%m"),"validation_months":[idx.strftime("%Y-%m") for idx in test.index],"latest_historical_revenue":float(complete.iloc[-1]),"next_month_forecast":float(fc.predicted_mean.iloc[0]),"next_three_months_forecast_total":float(fc.predicted_mean.sum())}
    return metrics,pd.DataFrame(rows),final,pd.DataFrame({"month":test.index.strftime("%Y-%m"),"actual":test.to_numpy(),"backtest_forecast":pred.to_numpy()})
