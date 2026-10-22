"""
Explainable AI add-on for the ToN-IoT experiment.
Trains XGBoost with the settings selected in the main experiment, then explains it with
SHAP (global and per-class) and permutation importance.

Colab setup (run in a cell first):
    !pip install xgboost shap

Keep SAMPLE and SEED the same as in toniot_experiment.py so the test set is identical.

Outputs: shap_summary_binary.png, shap_bar_binary.png, permutation_importance_binary.png,
         shap_bar_multiclass.png, shap_per_class_top5.csv
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ---------------- settings ----------------
CSV_PATH = "Train_Test_Network.csv"
OUT_DIR = "."
SEED = 42
SAMPLE = 150000          # must match the main experiment
MAX_CATEGORIES = 30
N_SHAP = 2000            # test flows to explain (SHAP is slow on big samples)
N_PERM = 8000            # test flows used for permutation importance
# XGBoost settings selected by the search in the main experiment
PARAMS_BINARY = dict(n_estimators=100, max_depth=6, learning_rate=0.1)
PARAMS_MULTI = dict(n_estimators=300, max_depth=4, learning_rate=0.2)
# ------------------------------------------

os.makedirs(OUT_DIR, exist_ok=True)
ID_COLS = ["ts", "src_ip", "dst_ip", "src_port"]

# ---------- same preprocessing as the main experiment ----------
df = pd.read_csv(CSV_PATH, low_memory=False)
df = df.replace("-", np.nan).drop_duplicates()
if SAMPLE and len(df) > SAMPLE:
    df = df.sample(SAMPLE, random_state=SEED)
y_binary = df["label"].astype(int).values
y_type = df["type"].astype(str).values
X = df.drop(columns=["label", "type"] + [c for c in ID_COLS if c in df.columns])


def is_text(col):
    return not pd.api.types.is_numeric_dtype(col)


for c in X.columns:
    if is_text(X[c]):
        conv = pd.to_numeric(X[c], errors="coerce")
        if conv.notna().sum() > 0.9 * X[c].notna().sum() and X[c].notna().sum() > 0:
            X[c] = conv
cat_cols = [c for c in X.columns if is_text(X[c])]
dropped = [c for c in cat_cols if X[c].nunique() > MAX_CATEGORIES]
cat_cols = [c for c in cat_cols if c not in dropped]
X = X.drop(columns=dropped)
for c in cat_cols:
    X[c] = X[c].fillna("none").astype(str)
X = pd.get_dummies(X, columns=cat_cols).fillna(0).astype(float)
# XGBoost refuses column names containing [ ] or <, which can appear in category values
X.columns = [str(c).replace("[", "(").replace("]", ")").replace("<", "lt_") for c in X.columns]
if X.columns.duplicated().any():
    X.columns = [f"{c}_{i}" if d else c for i, (c, d) in enumerate(zip(X.columns, X.columns.duplicated(keep=False)))]
print("Feature matrix:", X.shape)


def fit_xgb(y_raw, params):
    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    model = XGBClassifier(random_state=SEED, n_jobs=-1, tree_method="hist", **params).fit(X_tr, y_tr)
    print("Test accuracy:", round(float((model.predict(X_te) == y_te).mean()), 4))
    return model, X_te, y_te, list(enc.classes_)


# ---------- binary task ----------
model, X_te, y_te, classes = fit_xgb(y_binary, PARAMS_BINARY)
sample = X_te.sample(min(N_SHAP, len(X_te)), random_state=SEED)
sv = shap.TreeExplainer(model).shap_values(sample)

plt.figure()
shap.summary_plot(sv, sample, max_display=15, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_summary_binary.png"), dpi=200, bbox_inches="tight")
plt.close()

plt.figure()
shap.summary_plot(sv, sample, plot_type="bar", max_display=15, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar_binary.png"), dpi=200, bbox_inches="tight")
plt.close()

# permutation importance: how much does F1 drop when one feature is shuffled?
sub = X_te.sample(min(N_PERM, len(X_te)), random_state=SEED)
perm = permutation_importance(model, sub, y_te[X_te.index.get_indexer(sub.index)],
                              scoring="f1", n_repeats=5, random_state=SEED, n_jobs=-1)
pi = pd.DataFrame({"feature": X.columns, "mean": perm.importances_mean, "std": perm.importances_std})
pi = pi.sort_values("mean", ascending=False).head(15)
plt.figure(figsize=(8, 6))
plt.barh(pi["feature"][::-1], pi["mean"][::-1], xerr=pi["std"][::-1], color="steelblue")
plt.xlabel("Drop in F1 when the feature is shuffled")
plt.title("Permutation importance, XGBoost (binary)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "permutation_importance_binary.png"), dpi=200)
plt.close()
print("\nTop permutation importances:\n", pi.round(4).to_string(index=False))

# ---------- ten-class task ----------
model, X_te, y_te, classes = fit_xgb(y_type, PARAMS_MULTI)
sample = X_te.sample(min(N_SHAP, len(X_te)), random_state=SEED)
sv = shap.TreeExplainer(model).shap_values(sample)
arr = np.stack(sv, axis=-1) if isinstance(sv, list) else sv      # shape (n, features, classes)
mean_abs = np.abs(arr).mean(axis=0)                               # (features, classes)

overall = pd.Series(mean_abs.mean(axis=1), index=X.columns).sort_values(ascending=False).head(15)
plt.figure(figsize=(8, 6))
overall[::-1].plot(kind="barh", color="seagreen")
plt.xlabel("Mean |SHAP value| averaged over classes")
plt.title("Global feature impact, XGBoost (ten-class)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "shap_bar_multiclass.png"), dpi=200)
plt.close()

rows = []
for j, name in enumerate(classes):
    top = pd.Series(mean_abs[:, j], index=X.columns).sort_values(ascending=False).head(5)
    rows.append({"class": name, **{f"top{k+1}": f"{f} ({v:.3f})" for k, (f, v) in enumerate(top.items())}})
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "shap_per_class_top5.csv"), index=False)
print("\nTop features per class:\n", pd.DataFrame(rows).to_string(index=False))
print("\nDone. Files written to", OUT_DIR)
