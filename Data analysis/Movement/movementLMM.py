import os
import re
import ast
import json
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf

MISSING_JSON = "../missingAnimals.json"
MOVEMENT_DIR = "../output/movement"  
TJ_LIST = ["0_2", "2_7", "7_9", "9_17"]  

def parse_tj(tj: str):
    """Return start_day, end_day, days_in_period from a string like '0_7'."""
    a, b = map(int, tj.split("_"))
    return a, b, b - a
    
def normalize(s):
    return str(s).strip().casefold()
    
def period_label(tj: str) -> str:
    """Optional human readable label."""
    a, b, _ = parse_tj(tj)
    return f"D{a}-D{b}"

with open(MISSING_JSON, "r") as f:
    missing_animals = json.load(f)

def treatment_for_exp(exp_id: int) -> str:
    return "Pheromone" if exp_id in [2, 4, 5, 8] else "Control"

def prefix_for_exp(exp_id: int) -> str:
    return "ExperimentRemux" if exp_id == 8 else "Experiment"

rows = []

for exp_id in range(2, 10):
    prefix = prefix_for_exp(exp_id)
    exp_folder = os.path.join(MOVEMENT_DIR, f"{prefix}{exp_id}")

    missing_list = missing_animals[str(exp_id)]
    animal_count = 21 - len(missing_list)

    treat = treatment_for_exp(exp_id)

    for tj in TJ_LIST:
        txt_path = os.path.join(exp_folder, f"{tj}.txt")
        if not os.path.exists(txt_path):
            continue

        with open(txt_path, "r") as f:
            data_dict = ast.literal_eval(f.read())

        start_day, end_day, days_in_period = parse_tj(tj)
        per_label = period_label(tj)

        for sow_id, outcome in data_dict.items():
            rows.append({
                "exp_id": exp_id,
                "treatment": treat,
                "tJ": tj,
                "period_label": per_label,
                "period_start_day": start_day,
                "period_end_day": end_day,
                "days_in_period": days_in_period,
                "group_size": animal_count,
                "sow_id": str(sow_id),
                "outcome": outcome,   
            })

df = pd.DataFrame(rows).sort_values(["exp_id", "tJ", "sow_id"]).reset_index(drop=True)

raw = """
2 3 2 Dimi 16 1
2 3 2 Dimi 17 2
2 3 2 Dimi 23 3
2 3 2 Dimi 27 4
2 3 2 Dimi 43 5
2 3 2 Dimi 111 6
2 3 2 Dimi 119 7
2 3 2 Dimi 132 8
2 3 2 Dimi 134 9
2 3 2 Dimi 138 10
2 3 2 Dimi 150 11
2 3 2 Dimi 154 12
2 3 2 Dimi 156 13
2 3 2 Dimi 160 14
2 3 2 Dimi 167 15
2 3 2 Dimi 179 16
3 2 2 Dimi 2 1
3 2 2 Dimi 5 2
3 2 2 Dimi 6 3
3 2 2 Dimi 18 4
3 2 2 Dimi 25 5
3 2 2 Dimi 38 6
3 2 2 Dimi 62 7
3 2 2 Dimi 70 8
3 2 2 Dimi 92 9
3 2 2 Dimi 98 10
3 2 2 Dimi 106 11
3 2 2 Dimi 117 12
3 2 2 Dimi 121 13
3 2 2 Dimi 126 14
3 2 2 Dimi 130 16
3 2 2 Dimi 133 17
3 2 2 Dimi 137 18
3 2 2 Dimi 143 19
3 2 2 Dimi 175 20
3 2 2 Dimi 39 21
4 3 2 Dimi 1 1
4 3 2 Dimi 4 2
4 3 2 Dimi 19 3
4 3 2 Dimi 30 5
4 3 2 Dimi 41 6
4 3 2 Dimi 42 7
4 3 2 Dimi 68 8
4 3 2 Dimi 77 9
4 3 2 Dimi 81 10
4 3 2 Dimi 95 11
4 3 2 Dimi 100 12
4 3 2 Dimi 109 13
4 3 2 Dimi 110 14
4 3 2 Dimi 122 15
4 3 2 Dimi 139 16
4 3 2 Dimi 148 17
4 3 2 Dimi 164 18
4 3 2 Dimi 170 19
4 3 2 Dimi 186 20
5 2 2 Dimi 16 1
5 2 2 Dimi 17 2
5 2 2 Dimi 23 3
5 2 2 Dimi 27 4
5 2 2 Dimi 43 5
5 2 2 Dimi 55 6
5 2 2 Dimi 58 7
5 2 2 Dimi 61 8
5 2 2 Dimi 63 9
5 2 2 Dimi 66 10
5 2 2 Dimi 150 11
5 2 2 Dimi 179 12
5 2 2 Dimi 156 13
5 2 2 Dimi 160 14
5 2 2 Dimi 163 15
5 2 2 Dimi 154 16
6 3 2 Dimi 7 1
6 3 2 Dimi 11 2
6 3 2 Dimi 20 3
6 3 2 Dimi 22 4
6 3 2 Dimi 28 5
6 3 2 Dimi 29 6
6 3 2 Dimi 53 8
6 3 2 Dimi 90 9
6 3 2 Dimi 94 10
6 3 2 Dimi 109 11
6 3 2 Dimi 111 12
6 3 2 Dimi 118 13
6 3 2 Dimi 119 14
6 3 2 Dimi 146 15
6 3 2 Dimi 153 16
6 3 2 Dimi 47 17
6 3 2 Dimi 49 18
7 2 2 Lisa 19 1
7 2 2 Lisa 21 2
7 2 2 Lisa 30 3
7 2 2 Lisa 41 4
7 2 2 Lisa 42 5
7 2 2 Lisa 50 6
7 2 2 Lisa 68 7
7 2 2 Lisa 77 8
7 2 2 Lisa 81 9
7 2 2 Lisa 88 10
7 2 2 Lisa 100 11
7 2 2 Lisa 110 12
7 2 2 Lisa 122 13
7 2 2 Lisa 148 14
7 2 2 Lisa 164 15
7 2 2 Lisa 168 16
7 2 2 Lisa 170 17
7 2 2 Lisa 176 18
8 1 2 Lisa 16 1
8 1 2 Lisa 17 2
8 1 2 Lisa 23 3
8 1 2 Lisa 27 4
8 1 2 Lisa 43 5
8 1 2 Lisa 55 6
8 1 2 Lisa 58 7
8 1 2 Lisa 61 8
8 1 2 Lisa 63 9
8 1 2 Lisa 95 10
8 1 2 Lisa 113 11
8 1 2 Lisa 131 12
8 1 2 Lisa 134 13
8 1 2 Lisa 136 14
8 1 2 Lisa 150 15
8 1 2 Lisa 156 16
8 1 2 Lisa 157 17
8 1 2 Lisa 163 18
9 2 2 Lisa 7 1
9 2 2 Lisa 11 2
9 2 2 Lisa 20 3
9 2 2 Lisa 28 4
9 2 2 Lisa 29 5
9 2 2 Lisa 33 6
9 2 2 Lisa 45 7
9 2 2 Lisa 52 8
9 2 2 Lisa 53 9
9 2 2 Lisa 94 10
9 2 2 Lisa 103 11
9 2 2 Lisa 109 12
9 2 2 Lisa 111 13
9 2 2 Lisa 119 14
9 2 2 Lisa 146 15
9 2 2 Lisa 153 16
9 2 2 Lisa 155 17
"""

rows = []
for line in raw.splitlines():
    line = line.strip()
    if not line:
        continue
    # split on whitespace
    parts = re.split(r"\s+", line)
    if len(parts) < 6:
        continue
    try:
        exp_id = int(parts[0])
        sow_actual = parts[4]
        id_in_exp = int(parts[5])
    except ValueError:
        continue
    rows.append((exp_id, sow_actual, id_in_exp))

map_df = pd.DataFrame(rows, columns=["exp_id", "sow_actual_id", "id_in_experiment"])
map_dict = {(r.exp_id, r.sow_actual_id): r.id_in_experiment for r in map_df.itertuples(index=False)}
name_col = 'sow_id'
map_df = map_df.copy()
map_df['__key'] = map_df['exp_id'].astype(int).astype(str) + '|' + map_df['id_in_experiment'].map(normalize)

df = df.copy()
df['__key'] = df['exp_id'].astype(int).astype(str) + '|' + df[name_col].map(normalize)

lookup = pd.Series(map_df['sow_actual_id'].values, index=map_df['__key']).to_dict()
df['sow_uid'] = df['__key'].map(lookup)
df = df.drop(columns='__key')




period_order = ['D0-D2','D2-D7','D7-D9','D9-D17']
df['period'] = pd.Categorical(df['period_label'], categories=period_order, ordered=True)
df['outcomeNorm'] = df['outcome'] / df['days_in_period']

# Fit interaction model
res_periods = smf.mixedlm("outcomeNorm ~ C(treatment) * C(period)",
                          data=df, groups=df["sow_uid"],
                          vc_formula={"exp": "0 + C(exp_id)"},
                          re_formula="1").fit(reml=True)

def parse_days(lbl):
    a, b = lbl.replace('D','').split('-')
    return int(a), int(b)

params, V = res_periods.params, res_periods.cov_params()
names = params.index.tolist()

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

vc = {"exp": "0 + C(exp_id)"}
m_overall = smf.mixedlm("outcome ~ C(treatment) + C(period)",
                        data=df, groups=df["sow_uid"],
                        vc_formula=vc, re_formula="1")
res_overall = m_overall.fit(reml=True)

b = res_overall.params['C(treatment)[T.Pheromone]']
se = res_overall.bse['C(treatment)[T.Pheromone]']
z  = b / se
p  = 2*(1 - norm.cdf(abs(z)))
row_overall_reduced = dict(period='Pooled', start_day=0, end_day=17,
                           n_days=17, estimate=b, se=se, ci_lo=b-1.96*se, ci_hi=b+1.96*se,
                           p=p, p_holm=np.nan)

w = (df.groupby('period', observed=True).size()
       .reindex(period_order).fillna(0).to_numpy(dtype=float))
w = w / w.sum()

c_overall = np.zeros(len(names))
for wp, per in zip(w, period_order):
    c = np.zeros(len(names))
    terms = ['C(treatment)[T.Pheromone]']
    if per != period_order[0]:
        terms.append(f'C(treatment)[T.Pheromone]:C(period)[T.{per}]')
    for t in terms:
        if t in names:
            c[names.index(t)] += 1.0
    c_overall += wp * c


tbl_all = pd.concat([tbl, pd.DataFrame([row_overall_reduced])], ignore_index=True)
print(tbl_all[['period','start_day','end_day','n_days','estimate','se','ci_lo','ci_hi','p']].to_string(index=False))
