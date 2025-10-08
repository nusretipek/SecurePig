import pandas as pd
import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf


df = pd.read_csv("WQSows.csv")
period_order = ['D0-D2','D2-D7','D7-D9','D9-D17']
df['period'] = pd.Categorical(df['Period label'], categories=period_order, ordered=True)
df['treatment'] = df["Pheromone"]
df['outcome'] = df['Lesion score total'].astype(int) 
df['sow_uid'] = df['Sow earmark'].astype(int) 
df['exp_id'] = df['Round'].astype(int) 

# Treatment Effect model
res_treatment = smf.mixedlm("outcome ~ C(treatment) + C(period)",
                            data=df,
                            groups=df["sow_uid"],
                            vc_formula={"exp": "0 + C(exp_id)"},
                            re_formula="1").fit(method="lbfgs", reml=True, maxiter=5000)

params = res_treatment.params
V = res_treatment.cov_params()
names = params.index.tolist()

def treatment_effect():
    c = np.zeros(len(names))
    tname = 'C(treatment)[T.Pheromone]'
    if tname in names:
        c[names.index(tname)] = 1.0

    est = float(c @ params.values)
    se  = float(np.sqrt(c @ V.values @ c))
    z   = est / se
    p   = 2*(1 - norm.cdf(abs(z)))

    return dict(estimate=est,
                se=se,
                ci_lo=est-1.96*se,
                ci_hi=est+1.96*se,
                z=z,
                p=p)

tbl_treatment = pd.DataFrame([treatment_effect()])

# Period-specific model
res_periods = smf.mixedlm("outcome ~ C(treatment) * C(period)",
                          data=df, groups=df["sow_uid"],
                          vc_formula={"exp": "0 + C(exp_id)"},
                          re_formula="1").fit(method="lbfgs", reml=True, maxiter=5000)

params, V = res_periods.params, res_periods.cov_params()
names = params.index.tolist()

def parse_days(lbl):
    a, b = lbl.replace('D','').split('-')
    return int(a), int(b)

def contrast_period(per):
    c = np.zeros(len(names))
    terms = ['C(treatment)[T.Pheromone]']
    if per != period_order[0]:
        terms.append(f'C(treatment)[T.Pheromone]:C(period)[T.{per}]')
    for t in terms:
        if t in names:
            c[names.index(t)] += 1.0
    est = float(c @ params.values)
    se  = float(np.sqrt(c @ V.values @ c))
    z   = est / se
    p   = 2*(1 - norm.cdf(abs(z)))
    d0, d1 = parse_days(per)
    return dict(period=per, start_day=d0, end_day=d1, n_days=d1-d0,
                estimate=est, se=se, ci_lo=est-1.96*se, ci_hi=est+1.96*se, p=p)

tbl = pd.DataFrame([contrast_period(p) for p in period_order])

# Print results
tbl_treatment.insert(0, "period", ["Pooled"])
tbl_treatment["start_day"] = 0
tbl_treatment["end_day"] = 17
tbl_treatment["n_days"] = 17
tbl_treatment = tbl_treatment[["period","start_day","end_day","n_days",
                               "estimate","se","ci_lo","ci_hi","p"]]
tbl_merged = pd.concat([tbl, tbl_treatment], ignore_index=True)

# Print nicely
print(tbl_merged.to_string(index=False))