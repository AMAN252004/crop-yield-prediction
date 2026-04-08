# src/auto_preprocess.py
import re
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_PATH = "/Users/amanjakhar/Desktop/soybean_data_soilgrid250_modified_states_9.csv"  # <--- your file
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH)
print("Rows,Cols:", df.shape)

cols = list(df.columns)
print("\n--- first 20 cols ---")
print(cols[:20])
print("\nTotal columns:", len(cols))

# 1) find W_ groups (weather)
w_pattern = re.compile(r"^W_(\d+)_(\d+)$")
weather_coords = []
for c in cols:
    m = w_pattern.match(c)
    if m:
        var_idx = int(m.group(1))
        week_idx = int(m.group(2))
        weather_coords.append((c, var_idx, week_idx))
if weather_coords:
    # Build dict var_idx -> list of (week_idx, column)
    vars_dict = {}
    for col, v, w in weather_coords:
        vars_dict.setdefault(v, []).append((w, col))
    # sort weeks per var
    for v in vars_dict:
        vars_dict[v].sort(key=lambda x: x[0])
    var_ids_sorted = sorted(vars_dict.keys())
    num_vars = len(var_ids_sorted)
    num_weeks = max(max(w for w,_ in vars_dict[v]) for v in vars_dict)
    print(f"\nDetected Weather: {num_vars} variables, {num_weeks} weeks (max week index).")
    for v in var_ids_sorted:
        print(f" W var {v}: {len(vars_dict[v])} week columns (example: {vars_dict[v][0][1]} ...)")
else:
    print("\nNo W_ columns found with pattern W_{var}_{week}.")

# 2) find P_ management columns
p_pattern = re.compile(r"^P_(\d+)$")
p_cols = sorted([c for c in cols if p_pattern.match(c)], key=lambda x: int(p_pattern.match(x).group(1)))
if p_cols:
    print(f"\nDetected {len(p_cols)} management P_ columns (first/last): {p_cols[0]} ... {p_cols[-1]}")
else:
    print("\nNo P_ management columns found.")

# 3) detect soil columns heuristically: common soil acronyms or S_ prefix
soil_prefixes = ['bdod','cec','cfvo','clay','nitrogen','ocd','ocs','phh2o','sand','silt','soc']
soil_cols = []
for c in cols:
    lc = c.lower()
    # depth suffix like _0_5 or _0-5
    if any(lc.startswith(pref) for pref in soil_prefixes) or lc.startswith('s_') or re.match(r"^S_\w+", c):
        soil_cols.append(c)
# If none found via acronyms, also search for patterns like 'soil' or 'depth'
if not soil_cols:
    soil_cols = [c for c in cols if 'soil' in c.lower() or 'depth' in c.lower()]
print(f"\nDetected soil columns: {len(soil_cols)} (show up to 10):", soil_cols[:10])

# 4) Identify target and id cols
target_col = None
for possible in ['yield','Yield','YIELD','yield_bu_acre','y']:
    if possible in df.columns:
        target_col = possible
        break
if not target_col:
    # heuristics: pick column named exactly 'yield' from your earlier print -> you said 'yield'
    if 'yield' in df.columns:
        target_col = 'yield'
print("\nTarget column:", target_col)

id_col = None
for possible in ['loc_ID','loc_id','location','id','county_fips','FIPS']:
    if possible in df.columns:
        id_col = possible
        break
print("ID column:", id_col)

# === Build X_weather matrix if possible ===
def build_weather_matrix(df, vars_dict):
    # vars_dict: v -> list of (week, col) sorted by week
    var_list = sorted(vars_dict.keys())
    N = df.shape[0]
    max_week = max(max(w for w,_ in vars_dict[v]) for v in vars_dict)
    # find actual number of weeks per var (if some missing we pad with nan)
    week_counts = [len(vars_dict[v]) for v in var_list]
    max_weeks = max(week_counts)
    Xw = np.full((N, max_weeks, len(var_list)), np.nan, dtype=float)
    for j,v in enumerate(var_list):
        weeks_cols = vars_dict[v]  # list of (week, col) sorted
        for i_w, col in enumerate([c for _,c in weeks_cols]):
            Xw[:, i_w, j] = df[col].values
    return Xw, var_list

Xw = None
if weather_coords:
    Xw, var_list = build_weather_matrix(df, vars_dict)
    print("X_weather shape (N, weeks, vars):", Xw.shape)

# === Build soil array (N, features) or (N, depths, vars) ===
if soil_cols:
    # naive: keep as flat vector of soil columns
    Xs = df[soil_cols].values.astype(float)
    print("X_soil shape (N, soil_features):", Xs.shape)
else:
    Xs = None

# === Build mgmt array ===
if p_cols:
    Xm = df[p_cols].values.astype(float)
    print("X_mgmt shape (N, mgmt_features):", Xm.shape)
else:
    Xm = None

# === Target y ===
if target_col:
    y = df[target_col].values.astype(float)
    print("y shape:", y.shape)
else:
    raise RuntimeError("No target column found - cannot proceed.")

# === Simple imputation and scaling for soil and mgmt ===
def impute_and_scale(train_arr):
    arr = np.array(train_arr, dtype=float)
    # impute nan with column mean
    col_mean = np.nanmean(arr, axis=0)
    inds = np.where(np.isnan(arr))
    if inds[0].size:
        arr[inds] = np.take(col_mean, inds[1])
    scaler = StandardScaler()
    arr_scaled = scaler.fit_transform(arr)
    return arr_scaled, scaler

# impute/scale soil & mgmt
scalers = {}
if Xs is not None:
    Xs_scaled, scalers['soil'] = impute_and_scale(Xs)
else:
    Xs_scaled = None
if Xm is not None:
    Xm_scaled, scalers['mgmt'] = impute_and_scale(Xm)
else:
    Xm_scaled = None

# For weather: fill nan with weekly variable mean (axis=0 -> over samples)
if Xw is not None:
    Xw = np.array(Xw, dtype=float)
    # Xw shape (N, weeks, vars). Fill nan with column means per week-var
    # flatten to (N, weeks*vars) to compute means and impute quickly
    flat = Xw.reshape(Xw.shape[0], -1)
    col_mean = np.nanmean(flat, axis=0)
    inds = np.where(np.isnan(flat))
    if inds[0].size:
        flat[inds] = np.take(col_mean, inds[1])
    Xw = flat.reshape(Xw.shape)

# Save arrays
np.save(os.path.join(OUT_DIR, "Xw.npy"), Xw)
np.save(os.path.join(OUT_DIR, "Xs.npy"), Xs_scaled)
np.save(os.path.join(OUT_DIR, "Xm.npy"), Xm_scaled)
np.save(os.path.join(OUT_DIR, "y.npy"), y)
# save metadata
import json
meta = {
    "weather_var_ids": var_list if weather_coords else [],
    "p_cols": p_cols,
    "soil_cols": soil_cols,
    "target_col": target_col,
    "id_col": id_col
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)

print("\nSaved processed arrays to", OUT_DIR)
print("Meta summary:", meta)