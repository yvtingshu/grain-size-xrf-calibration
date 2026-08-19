"""
De-identified public release.

This script preserves the analysis/modeling workflow while removing direct
references to internal institutions, projects, stations/cores, and source-file
naming conventions. Do not distribute confidential source data with this code.
Use synthetic or otherwise approved de-identified example data for public use.
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

#==
# 0. Input data configuration
#==
# Public-release notes:
# 1) Project, institution, site/core identifiers, and internal filename conventions are excluded.
# 2) Place approved de-identified files at the generic paths below, or update the paths as needed.
# 3) Public repositories should contain only synthetic/example data, not confidential source data.
DATA_FILES = {
    "dataset_a": {
        "scan": "data/dataset_a_scan.xlsx",
        "chem": "data/dataset_a_chem.xlsx",
        "grain": "data/dataset_a_grain.xlsx",
        "sheet_name": "Sheet1",
    },
    "dataset_b": {
        "scan": "data/dataset_b_scan.xlsx",
        "chem": "data/dataset_b_chem.xlsx",
        "grain": "data/dataset_b_grain.xlsx",
        "sheet_name": "Sheet1",
    },
}

def require_file(path):
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Required input file not found: {path}. "
            "Use de-identified or synthetic example data only."
        )

for cfg in DATA_FILES.values():
    for key in ("scan", "chem", "grain"):
        require_file(cfg[key])

#==
# 1. Load data and standardize column names
#==
def robust_rename(df):
    df.columns = [c.replace('（', '(').replace('）', ')').strip() for c in df.columns]
    station_col = None
    for col in df.columns:
        if any(kw in col for kw in ['站号', '站位编号', 'Station']):
            station_col = col
            break
    if station_col is None:
        station_col = df.columns[0]
    df.rename(columns={station_col: 'Station'}, inplace=True)
    top_col, bot_col = None, None
    for col in df.columns:
        if '起始深度' in col or '顶深' in col:
            top_col = col
        elif '终止深度' in col or '底深' in col:
            bot_col = col
    if top_col is None:
        top_col = df.columns[1]
    if bot_col is None:
        bot_col = df.columns[2]
    df.rename(columns={top_col: 'Top_cm', bot_col: 'Bot_cm'}, inplace=True)
    df['Top_cm'] = pd.to_numeric(df['Top_cm'], errors='coerce')
    df['Bot_cm'] = pd.to_numeric(df['Bot_cm'], errors='coerce')
    df['Center_cm'] = (df['Top_cm'] + df['Bot_cm']) / 2.0
    return df

scan_a = robust_rename(pd.read_excel(
    DATA_FILES["dataset_a"]["scan"], sheet_name=DATA_FILES["dataset_a"]["sheet_name"]))
chem_a = robust_rename(pd.read_excel(
    DATA_FILES["dataset_a"]["chem"], sheet_name=DATA_FILES["dataset_a"]["sheet_name"]))
grain_a = robust_rename(pd.read_excel(
    DATA_FILES["dataset_a"]["grain"], sheet_name=DATA_FILES["dataset_a"]["sheet_name"]))
scan_b = robust_rename(pd.read_excel(
    DATA_FILES["dataset_b"]["scan"], sheet_name=DATA_FILES["dataset_b"]["sheet_name"]))
chem_b = robust_rename(pd.read_excel(
    DATA_FILES["dataset_b"]["chem"], sheet_name=DATA_FILES["dataset_b"]["sheet_name"]))
grain_b = robust_rename(pd.read_excel(
    DATA_FILES["dataset_b"]["grain"], sheet_name=DATA_FILES["dataset_b"]["sheet_name"]))
if '灼失量(%)' in chem_a.columns:
    chem_a.rename(columns={'灼失量(%)': 'Loss On(%)'}, inplace=True)

#==
# 2. Standardize grain-size column names
#==
GRAIN_COLUMN_MAP_A = {
    '砾石含量(%)': 'Gravel', '砂含量(%)': 'Sand', '粉砂含量(%)': 'Silt',
    '黏土含量(%)': 'Clay', '平均粒径(φ)': 'Mean_phi', '中值粒径(φ)': 'Median_phi',
    '分选系数': 'Sorting', '偏态': 'Skewness', '峰态': 'Kurtosis', '福克分类': 'Folk_class'
}
GRAIN_COLUMN_MAP_B = {
    '砾石/%': 'Gravel', '砂/%': 'Sand', '粉砂/%': 'Silt', '粘土/%': 'Clay',
    'MZ/φ': 'Mean_phi', 'sigma/φ': 'Sorting', 'Ski': 'Skewness', 'Kg': 'Kurtosis', 'Md/μm': 'Median_phi'
}
grain_a.rename(columns=GRAIN_COLUMN_MAP_A, inplace=True)
grain_b.rename(columns=GRAIN_COLUMN_MAP_B, inplace=True)
STANDARD_GRAIN_COLUMNS = ['Gravel', 'Sand', 'Silt', 'Clay', 'Mean_phi', 'Median_phi', 'Sorting', 'Skewness', 'Kurtosis']

#==
# 3. Identify common XRF features and laboratory target columns
#==
BASE_COLUMNS = ['Station', 'Top_cm', 'Bot_cm', 'Center_cm']
COMMON_XRF_ELEMENTS = sorted(list(set([c for c in scan_a.columns if c not in BASE_COLUMNS]) &
                           set([c for c in scan_b.columns if c not in BASE_COLUMNS])))

TARGET_OXIDES = ['SiO2', 'Al2O3', 'Fe2O3', 'CaO', 'MgO', 'K2O', 'Na2O', 'TiO2', 'MnO', 'P2O5']

def find_lab(df, elem):
    candidates = [c for c in df.columns if c.startswith('Lab_') and elem in c]
    if not candidates:
        candidates = [c for c in df.columns if elem in c and c not in BASE_COLUMNS]
    if not candidates:
        raise ValueError(f"No laboratory column found for target: {elem}")
    return candidates[0]

#==
# 4. Build the combined dataset (raw XRF + CLR-transformed XRF + grain-size features)
#==
DATASET_CONFIGS = [
    {'name': 'Dataset_A', 'scan': scan_a, 'chem': chem_a, 'grain': grain_a,
     'half_window': 1.0, 'expected_points': 4},
    {'name': 'Dataset_B', 'scan': scan_b, 'chem': chem_b, 'grain': grain_b,
     'half_window': 0.5, 'expected_points': 5}
]

all_records = []
for config in DATASET_CONFIGS:
    df_scan = config['scan']; df_chem = config['chem']; df_grain = config['grain']
    half_win = config['half_window']; dataset_name = config['name']
    for station in df_chem['Station'].unique():
        scan_st = df_scan[df_scan['Station'] == station].sort_values('Center_cm')
        chem_st = df_chem[df_chem['Station'] == station].sort_values('Center_cm')
        grain_st = df_grain[df_grain['Station'] == station].sort_values('Center_cm')
        chem_st = chem_st.merge(grain_st[['Station', 'Top_cm', 'Bot_cm'] + STANDARD_GRAIN_COLUMNS],
                                on=['Station', 'Top_cm', 'Bot_cm'], how='left')
        for _, chem_row in chem_st.iterrows():
            center = chem_row['Center_cm']
            mask = (scan_st['Center_cm'] >= center - half_win) & (scan_st['Center_cm'] <= center + half_win)
            window = scan_st.loc[mask]
            if len(window) < config['expected_points'] * 0.8:
                continue
            record = {'Station': station, 'Dataset': dataset_name}
            mean_vals = window[COMMON_XRF_ELEMENTS].mean()
            if mean_vals.isnull().any():
                continue
            # Raw XRF features
            for col in COMMON_XRF_ELEMENTS:
                record[f'XRF_raw_{col}'] = mean_vals[col]
            # CLR-transformed XRF features used in BL-2
            eps = 1e-12
            raw_vals = np.array([mean_vals[col] for col in COMMON_XRF_ELEMENTS])
            safe_vals = np.where(raw_vals <= 0, eps, raw_vals)
            log_vals = np.log(safe_vals)
            log_geom_mean = np.mean(log_vals)
            for i, col in enumerate(COMMON_XRF_ELEMENTS):
                record[f'XRF_clr_{col}'] = log_vals[i] - log_geom_mean
            # Grain-size features
            grain_vals = chem_row[STANDARD_GRAIN_COLUMNS]
            if grain_vals.isnull().any():
                continue
            for col in STANDARD_GRAIN_COLUMNS:
                record[f'Grain_{col}'] = grain_vals[col]
            # Laboratory reference values
            for col in df_chem.columns:
                if col not in BASE_COLUMNS + STANDARD_GRAIN_COLUMNS:
                    record[f'Lab_{col}'] = chem_row[col]
            all_records.append(record)

dataset = pd.DataFrame(all_records)
dataset.columns = [c.strip() for c in dataset.columns]

# Report the sample count for the current input data.
# The public version does not hard-code the confidential dataset sample count.
print(f"Total samples: {len(dataset)}")

TARGET_COLUMNS = [find_lab(dataset, e) for e in TARGET_OXIDES]
XRF_RAW_COLUMNS = [f'XRF_raw_{c}' for c in COMMON_XRF_ELEMENTS]
XRF_CLR_COLUMNS = [f'XRF_clr_{c}' for c in COMMON_XRF_ELEMENTS]
GRAIN_COLUMNS = [f'Grain_{c}' for c in STANDARD_GRAIN_COLUMNS]

ESSENTIAL_COLUMNS = XRF_RAW_COLUMNS + XRF_CLR_COLUMNS + GRAIN_COLUMNS + TARGET_COLUMNS
dataset = dataset.dropna(subset=ESSENTIAL_COLUMNS)
print(f"Samples after removing missing values: {len(dataset)}")

X_raw = dataset[XRF_RAW_COLUMNS].values.astype(np.float64)
X_clr = dataset[XRF_CLR_COLUMNS].values.astype(np.float64)
X_grain = dataset[GRAIN_COLUMNS].values.astype(np.float64)
X_full = np.hstack([X_raw, X_grain])
Y_all = dataset[TARGET_COLUMNS].values.astype(np.float64)

# Major-element XRF features used in the trace-element ablation experiment
MAJOR_XRF_ELEMENTS = ['Si', 'Al', 'K', 'Ca', 'Ti', 'Mn', 'Fe']
MAJOR_XRF_COLUMNS = [f'XRF_raw_{e}' for e in MAJOR_XRF_ELEMENTS if f'XRF_raw_{e}' in dataset.columns]
X_main = dataset[MAJOR_XRF_COLUMNS].values.astype(np.float64)
X_main_grain = np.hstack([X_main, X_grain])

#==
# 5. Model training functions
#==
def train_mlp_once(X_train, Y_train, X_test):
    """Train the bagged multi-output MLP on a given split and return predictions."""
    # Remove zero-variance features
    std_x = np.std(X_train, axis=0)
    nonzero = std_x > 1e-12
    X_train_f = X_train[:, nonzero]
    X_test_f = X_test[:, nonzero]
    # Feature selection: correlation threshold = 0.12; retain at least 15 features
    corr_threshold = 0.12
    min_features = 15
    selected_idx = []
    if X_train_f.shape[1] <= min_features:
        selected_idx = list(range(X_train_f.shape[1]))
    else:
        for i in range(X_train_f.shape[1]):
            max_corr = 0.0
            for j in range(Y_train.shape[1]):
                if np.std(Y_train[:, j]) == 0:
                    continue
                r, _ = pearsonr(X_train_f[:, i], Y_train[:, j])
                if abs(r) > max_corr:
                    max_corr = abs(r)
            if max_corr >= corr_threshold:
                selected_idx.append(i)
        if len(selected_idx) < min_features:
            all_corr = []
            for i in range(X_train_f.shape[1]):
                max_corr = 0.0
                for j in range(Y_train.shape[1]):
                    if np.std(Y_train[:, j]) == 0:
                        continue
                    r, _ = pearsonr(X_train_f[:, i], Y_train[:, j])
                    if abs(r) > max_corr:
                        max_corr = abs(r)
                all_corr.append(max_corr)
            selected_idx = np.argsort(all_corr)[-min_features:]
        selected_idx = sorted(selected_idx)
    X_train_sel = X_train_f[:, selected_idx]
    X_test_sel = X_test_f[:, selected_idx]
    # Z-score standardization of predictors
    scaler_x = StandardScaler().fit(X_train_sel)
    X_train_norm = scaler_x.transform(X_train_sel)
    X_test_norm = scaler_x.transform(X_test_sel)
    # Standardization of target variables
    y_scalers = []
    Y_train_norm = np.zeros_like(Y_train)
    for j in range(Y_train.shape[1]):
        sc = StandardScaler()
        Y_train_norm[:, j] = sc.fit_transform(Y_train[:, j].reshape(-1, 1)).ravel()
        y_scalers.append(sc)
    # Multi-output MLP with bagging
    base_mlp = MLPRegressor(hidden_layer_sizes=(18,), activation='tanh', solver='lbfgs',
                            alpha=1.0, max_iter=5000, random_state=42, tol=1e-6)
    multi_mlp = MultiOutputRegressor(base_mlp)
    bag = BaggingRegressor(multi_mlp, n_estimators=20, random_state=42, n_jobs=-1)
    bag.fit(X_train_norm, Y_train_norm)
    Y_pred_norm = bag.predict(X_test_norm)
    # Transform predictions back to the original target scale
    Y_pred = np.zeros_like(Y_pred_norm)
    for j in range(Y_train.shape[1]):
        Y_pred[:, j] = y_scalers[j].inverse_transform(Y_pred_norm[:, j].reshape(-1, 1)).ravel()
    return Y_pred

def train_linear_once(X_train, Y_train, X_test):
    """Train multivariate linear regression models on a given split and return predictions."""
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    Y_pred = np.zeros((X_test.shape[0], Y_train.shape[1]))
    for j in range(Y_train.shape[1]):
        lr = LinearRegression().fit(X_train_scaled, Y_train[:, j])
        Y_pred[:, j] = lr.predict(X_test_scaled)
    return Y_pred

#==
# 6. Relative deviation (RD%)
#==
def compute_rd(y_true, y_pred):
    """Compute mean absolute relative deviation (RD%)."""
    mask = np.abs(y_true) > 1e-12
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

#==
# 7. Repeated random train-test splits
#==
n_repeats = 10
n = len(X_full)

# Store performance metrics across repeated splits
bl1_results = {name: [] for name in TARGET_OXIDES}
bl2_results = {name: [] for name in TARGET_OXIDES}
bl3_results = {name: [] for name in TARGET_OXIDES}
bl4_results = {name: [] for name in TARGET_OXIDES}
no_grain_results = {name: [] for name in TARGET_OXIDES}
no_trace_results = {name: [] for name in TARGET_OXIDES}
rd_results = {name: [] for name in TARGET_OXIDES}

# Store out-of-sample predictions by sample index for RSD% calculation
# For each target, map each sample index to its out-of-sample predictions
sample_preds = {name: {i: [] for i in range(n)} for name in TARGET_OXIDES}

print("\nRunning 10 repeated random train-test splits...")
for rep in range(n_repeats):
    np.random.seed(rep)
    idx = np.random.permutation(n)
    split = n // 2
    train_idx, test_idx = idx[:split], idx[split:]
    
    Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]
    
    # BL-1: Raw XRF + linear regression
    Y_pred = train_linear_once(X_raw[train_idx], Y_train, X_raw[test_idx])
    for j, name in enumerate(TARGET_OXIDES):
        bl1_results[name].append(r2_score(Y_test[:, j], Y_pred[:, j]))
    
    # BL-2: CLR-transformed XRF + linear regression
    Y_pred = train_linear_once(X_clr[train_idx], Y_train, X_clr[test_idx])
    for j, name in enumerate(TARGET_OXIDES):
        bl2_results[name].append(r2_score(Y_test[:, j], Y_pred[:, j]))
    
    # BL-3: Raw XRF + MLP (without grain-size features)
    Y_pred = train_mlp_once(X_raw[train_idx], Y_train, X_raw[test_idx])
    for j, name in enumerate(TARGET_OXIDES):
        bl3_results[name].append(r2_score(Y_test[:, j], Y_pred[:, j]))
    
    # BL-4: Raw XRF + grain-size features + MLP (full feature set)
    Y_pred = train_mlp_once(X_full[train_idx], Y_train, X_full[test_idx])
    for j, name in enumerate(TARGET_OXIDES):
        bl4_results[name].append(r2_score(Y_test[:, j], Y_pred[:, j]))
        rd_results[name].append(compute_rd(Y_test[:, j], Y_pred[:, j]))
        # Record out-of-sample predictions by sample index
        for pos, sample_idx in enumerate(test_idx):
            sample_preds[name][sample_idx].append(Y_pred[pos, j])
    
    # Ablation 1: Remove grain-size features (XRF only)
    Y_pred = train_mlp_once(X_raw[train_idx], Y_train, X_raw[test_idx])
    for j, name in enumerate(TARGET_OXIDES):
        no_grain_results[name].append(r2_score(Y_test[:, j], Y_pred[:, j]))
    
    # Ablation 2: Remove trace-element XRF features (major-element XRF + grain size)
    Y_pred = train_mlp_once(X_main_grain[train_idx], Y_train, X_main_grain[test_idx])
    for j, name in enumerate(TARGET_OXIDES):
        no_trace_results[name].append(r2_score(Y_test[:, j], Y_pred[:, j]))
    
    if (rep + 1) % 2 == 0:
        print(f"Completed split {rep + 1}/{n_repeats}")

print("All repeated splits completed.")

#==
# 7.1. Relative standard deviation (RSD%)
# RSD% is calculated for each sample from predictions obtained when that sample
# appears in the test set across repeated splits, then averaged across samples.
#==
rsd_results = {}
for name in TARGET_OXIDES:
    rsd_values = []
    for i in range(n):
        preds = sample_preds[name][i]  # Out-of-sample predictions for this sample
        if len(preds) >= 2:  # At least two predictions are required to compute RSD
            mean_p = np.mean(preds)
            std_p = np.std(preds, ddof=1)  # Sample standard deviation
            if abs(mean_p) > 1e-12:
                rsd_values.append((std_p / abs(mean_p)) * 100)
    rsd_results[name] = np.mean(rsd_values) if rsd_values else 0.0

#==
# 8. Report baseline model performance
#==
print("\n" + "="*120)
print("Baseline Model Performance (mean ± SD across 10 splits)")
print("="*120)
header = f"{'Target':<8} {'BL-1 R²':<16} {'BL-2 R²':<16} {'BL-3 XRF-only R²':<18} {'BL-4 Full R²':<18} {'RD%':<10} {'RSD%':<10}"
print(header)
print("-"*100)

for name in TARGET_OXIDES:
    b1 = np.mean(bl1_results[name]); s1 = np.std(bl1_results[name])
    b2 = np.mean(bl2_results[name]); s2 = np.std(bl2_results[name])
    b3 = np.mean(bl3_results[name]); s3 = np.std(bl3_results[name])
    b4 = np.mean(bl4_results[name]); s4 = np.std(bl4_results[name])
    rd = np.mean(rd_results[name])
    rsd = rsd_results[name]
    print(f"{name:<8} {b1:.3f}±{s1:.3f}   {b2:.3f}±{s2:.3f}   {b3:.3f}±{s3:.3f}   {b4:.3f}±{s4:.3f}   {rd:.2f}    {rsd:.2f}")

print("\nNotes: BL-1 = raw XRF + linear regression; BL-2 = CLR-transformed XRF + linear regression;")
print("       BL-3 = raw XRF + MLP without grain-size features; BL-4 = XRF + grain-size features + MLP.")
print("       The difference ΔR² = BL-4 - BL-3 quantifies the contribution of grain-size features.")
print("       RD% and RSD% are calculated for BL-4 only.")
print("       RSD% is computed per sample from repeated out-of-sample predictions and then averaged across samples.")

#==
# 9. Report ablation-study performance
#==
print("\n" + "="*100)
print("MLP Ablation Study (mean ± SD across 10 splits)")
print("="*100)
print(f"{'Target':<8} {'Full R²':<16} {'No-grain R²':<16} {'No-trace R²':<16}")
print("-"*60)

for name in TARGET_OXIDES:
    full = np.mean(bl4_results[name]); full_std = np.std(bl4_results[name])
    no_grain = np.mean(no_grain_results[name]); no_grain_std = np.std(no_grain_results[name])
    no_trace = np.mean(no_trace_results[name]); no_trace_std = np.std(no_trace_results[name])
    print(f"{name:<8} {full:.3f}±{full_std:.3f}   {no_grain:.3f}±{no_grain_std:.3f}   {no_trace:.3f}±{no_trace_std:.3f}")

print("\nNotes: No-grain = XRF only; No-trace = major-element XRF (Si, Al, K, Ca, Ti, Mn, Fe) + grain-size features.")
print("       The ablation study uses the same 10 repeated random splits and reports mean ± SD.")