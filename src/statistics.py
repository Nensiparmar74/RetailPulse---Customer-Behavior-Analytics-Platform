from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, f_oneway

def run_statistical_tests(customers: pd.DataFrame, churn: pd.DataFrame, segmented: pd.DataFrame) -> list[dict]:
    """Run relevant t-test, chi-square, and ANOVA analyses and report hypotheses."""
    out=[]
    customer=customers.dropna(subset=["AOV"]).copy()
    uk=customer.loc[customer["Country"]=="United Kingdom","AOV"].dropna(); intl=customer.loc[customer["Country"]!="United Kingdom","AOV"].dropna()
    t,p=ttest_ind(uk,intl,equal_var=False,nan_policy="omit")
    out.append({"test_name":"Welch t-test: UK vs International AOV","null_hypothesis":"Mean AOV is equal between UK and international customers.","alternative_hypothesis":"Mean AOV differs between UK and international customers.","test_statistic":"t-statistic","statistic_value":float(t),"p_value":float(p),"significance_threshold":0.05,"is_statistically_significant":bool(p<0.05),"business_interpretation":f"At alpha=0.05, {'reject' if p<0.05 else 'do not reject'} H0. UK mean AOV={uk.mean():.2f}; international mean AOV={intl.mean():.2f}."})
    merged=segmented[["Customer ID","cluster"]].merge(churn[["Customer ID","churn"]],on="Customer ID",how="inner")
    table=pd.crosstab(merged["cluster"],merged["churn"]); chi,p,_,_=chi2_contingency(table)
    out.append({"test_name":"Chi-square: Segment vs churn status","null_hypothesis":"Customer segment and churn status are independent.","alternative_hypothesis":"Customer segment and churn status are associated.","test_statistic":"Chi-square","statistic_value":float(chi),"p_value":float(p),"significance_threshold":0.05,"is_statistically_significant":bool(p<0.05),"business_interpretation":f"At alpha=0.05, {'reject' if p<0.05 else 'do not reject'} H0; segment and churn status show {'an association' if p<0.05 else 'insufficient evidence of an association'}."})
    groups=[g["AOV"].dropna().to_numpy() for _,g in segmented.groupby("cluster")]
    f,p=f_oneway(*groups)
    out.append({"test_name":"One-way ANOVA: AOV across segments","null_hypothesis":"All segment mean AOVs are equal.","alternative_hypothesis":"At least one segment mean AOV differs.","test_statistic":"F-statistic","statistic_value":float(f),"p_value":float(p),"significance_threshold":0.05,"is_statistically_significant":bool(p<0.05),"business_interpretation":f"At alpha=0.05, {'reject' if p<0.05 else 'do not reject'} H0."})
    return out
