# =========================
# PAPER CORE (clean version)
# =========================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------
# Paths
# ---------
BASE_DIR = "/content/drive/MyDrive/paper_arm_vs_x86"
ARM_CSV  = f"{BASE_DIR}/arm_output.csv"
X86_XLSX = f"{BASE_DIR}/x86_output.xlsx"

# ----------------
# Helper functions
# ----------------
def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def cv_percent(x):
    x = pd.Series(x).dropna()
    if len(x) < 2 or x.mean() == 0:
        return np.nan
    return 100.0 * x.std(ddof=1) / x.mean()

def finite_positive(df, cols):
    out = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
    for c in cols:
        out = out[out[c] > 0]
    return out

# -------------
# 1) Load data
# -------------
df_arm = pd.read_csv(ARM_CSV)
df_x86 = pd.read_excel(X86_XLSX)

print("ARM rows:", df_arm.shape, "| x86 rows:", df_x86.shape)

# -----------------------------------------
# 2) Build RUN-level tables (aggregation)
# -----------------------------------------

# ===== ARM run-level =====
arm = df_arm.copy()

# numeric conversion (only columns we need)
for c in ["Total MFLOPS","Total Time","SPARSEMV MFLOPS","ampere","voltage",
          "nx","ny","nz","Number of MPI ranks","Avg DDOT MPI_Allreduce time",
          "Concluding timestamp","Starting run at"]:
    if c in arm.columns:
        arm[c] = to_num(arm[c])

# choose key
arm_key = None
if "Concluding timestamp" in arm.columns:
    arm_key = "Concluding timestamp"
elif "Starting run at" in arm.columns:
    arm_key = "Starting run at"
else:
    raise ValueError("ARM: no Concluding timestamp or Starting run at column found.")

# config cols (keep these if exist)
arm_config_cols = [c for c in ["Mini-Application Name","Mini-Application Version",
                               "Number of MPI ranks","nx","ny","nz","Number of iterations"]
                   if c in arm.columns]

arm_run = (
    arm.groupby([arm_key] + arm_config_cols, as_index=False)
       .agg(
           MFLOPS_total=("Total MFLOPS","mean"),
           MFLOPS_sparse=("SPARSEMV MFLOPS","mean") if "SPARSEMV MFLOPS" in arm.columns else ("Total MFLOPS","mean"),
           Time_total=("Total Time","mean"),
           ddot_allreduce_avg=("Avg DDOT MPI_Allreduce time","mean") if "Avg DDOT MPI_Allreduce time" in arm.columns else ("Total Time","mean"),
           ampere_mean=("ampere","mean"),
           voltage_mean=("voltage","mean"),
           n_samples=("Total Time","size")
       )
       .dropna(subset=["MFLOPS_total","Time_total","ampere_mean","voltage_mean","nx","ny","nz"])
)

# energy metrics
arm_run["Power_W"] = arm_run["ampere_mean"] * arm_run["voltage_mean"]
arm_run["Energy_J"] = arm_run["Power_W"] * arm_run["Time_total"]
arm_run["J_per_MFLOP"] = arm_run["Energy_J"] / arm_run["MFLOPS_total"]
arm_run["MFLOPS_per_W"] = arm_run["MFLOPS_total"] / arm_run["Power_W"]
arm_run["N"] = arm_run["nx"] * arm_run["ny"] * arm_run["nz"]

print("ARM runs:", arm_run.shape[0])

# ===== x86 run-level =====
x86 = df_x86.copy()

for c in ["MFLOPS total","Time total","MFLOPS SPARSEMV","nx","ny","nz","Unix time","Time"]:
    if c in x86.columns:
        x86[c] = to_num(x86[c])

x86_key = "Unix time" if "Unix time" in x86.columns else ("Time" if "Time" in x86.columns else None)
if x86_key is None:
    raise ValueError("x86: no Unix time or Time column found for run grouping.")

x86_run = (
    x86.groupby([x86_key, "nx", "ny", "nz"], as_index=False)
       .agg(
           MFLOPS_total=("MFLOPS total","mean"),
           MFLOPS_sparse=("MFLOPS SPARSEMV","mean") if "MFLOPS SPARSEMV" in x86.columns else ("MFLOPS total","mean"),
           Time_total=("Time total","mean"),
           n_samples=("MFLOPS total","size")
       )
       .dropna(subset=["MFLOPS_total","nx","ny","nz"])
)

x86_run["N"] = x86_run["nx"] * x86_run["ny"] * x86_run["nz"]
print("x86 runs:", x86_run.shape[0])

# -------------------------------------------------
# 3) Per-problem-size aggregation (paper tables)
# -------------------------------------------------

arm_ps = (
    arm_run.groupby(["nx","ny","nz","N"], as_index=False)
           .agg(
               runs_arm=("MFLOPS_total","size"),

               MFLOPS_arm_mean=("MFLOPS_total","mean"),
               MFLOPS_arm_std=("MFLOPS_total","std"),
               MFLOPS_arm_cv=("MFLOPS_total", cv_percent),

               Time_arm_mean=("Time_total","mean"),
               Time_arm_std=("Time_total","std"),
               Time_arm_cv=("Time_total", cv_percent),

               MFLOPSW_arm_mean=("MFLOPS_per_W","mean"),
               MFLOPSW_arm_std=("MFLOPS_per_W","std"),
               MFLOPSW_arm_cv=("MFLOPS_per_W", cv_percent),

               JperMFLOP_arm_mean=("J_per_MFLOP","mean"),
               JperMFLOP_arm_std=("J_per_MFLOP","std"),
               JperMFLOP_arm_cv=("J_per_MFLOP", cv_percent),
           )
           .sort_values("N")
)

x86_ps = (
    x86_run.groupby(["nx","ny","nz","N"], as_index=False)
           .agg(
               runs_x86=("MFLOPS_total","size"),

               MFLOPS_x86_mean=("MFLOPS_total","mean"),
               MFLOPS_x86_std=("MFLOPS_total","std"),
               MFLOPS_x86_cv=("MFLOPS_total", cv_percent),

               Time_x86_mean=("Time_total","mean"),
               Time_x86_std=("Time_total","std"),
               Time_x86_cv=("Time_total", cv_percent),
           )
           .sort_values("N")
)

print("ARM problem sizes:", arm_ps.shape[0])
print("x86 problem sizes:", x86_ps.shape[0])

# -----------------------------------------
# 4) Merge ARM vs x86 on same problem sizes
# -----------------------------------------
merged = pd.merge(arm_ps, x86_ps, on=["nx","ny","nz","N"], how="inner").sort_values("N")
print("Common problem sizes:", merged.shape[0])

merged["MFLOPS_ratio_x86_over_arm"] = merged["MFLOPS_x86_mean"] / merged["MFLOPS_arm_mean"]
merged["Stability_ratio_arm_over_x86"] = merged["MFLOPS_arm_cv"] / merged["MFLOPS_x86_cv"]

# -------------------------
# 5) Paper plots (core)
# -------------------------

# ARM: MFLOPS/W vs problem size
plt.figure()
plt.errorbar(arm_ps["N"], arm_ps["MFLOPSW_arm_mean"], yerr=arm_ps["MFLOPSW_arm_std"], fmt="o-", capsize=3)
plt.xscale("log")
plt.xlabel("Problem size N = nx·ny·nz (log)")
plt.ylabel("MFLOPS/W (mean ± std)")
plt.title("ARM: MFLOPS/W vs Problem Size")
plt.show()

# ARM: J/MFLOP vs problem size
plt.figure()
plt.errorbar(arm_ps["N"], arm_ps["JperMFLOP_arm_mean"], yerr=arm_ps["JperMFLOP_arm_std"], fmt="o-", capsize=3)
plt.xscale("log")
plt.xlabel("Problem size N = nx·ny·nz (log)")
plt.ylabel("J/MFLOP (mean ± std)")
plt.title("ARM: J/MFLOP vs Problem Size")
plt.show()

# ARM vs x86: MFLOPS vs problem size
plt.figure()
plt.plot(merged["N"], merged["MFLOPS_arm_mean"], "o-", label="ARM MFLOPS (mean)")
plt.plot(merged["N"], merged["MFLOPS_x86_mean"], "o-", label="x86 MFLOPS (mean)")
plt.xscale("log")
plt.xlabel("Problem size N = nx·ny·nz (log)")
plt.ylabel("MFLOPS (mean)")
plt.title("ARM vs x86: Performance vs Problem Size")
plt.legend()
plt.show()

# ARM vs x86: Stability (CV%) vs problem size
plt.figure()
plt.plot(merged["N"], merged["MFLOPS_arm_cv"], "o-", label="ARM CV% (MFLOPS)")
plt.plot(merged["N"], merged["MFLOPS_x86_cv"], "o-", label="x86 CV% (MFLOPS)")
plt.xscale("log")
plt.xlabel("Problem size N = nx·ny·nz (log)")
plt.ylabel("CV% (lower = more stable)")
plt.title("ARM vs x86: Stability vs Problem Size")
plt.legend()
plt.show()

# OPTIONAL runtime plot (only if all positive -> avoids log error)
rt = merged[["N","Time_arm_mean","Time_x86_mean"]].copy()
rt = finite_positive(rt, ["N","Time_arm_mean","Time_x86_mean"])
print("Runtime plot points after filtering:", rt.shape[0])
if rt.shape[0] > 0:
    plt.figure()
    plt.plot(rt["N"], rt["Time_arm_mean"], "o-", label="ARM Time (mean)")
    plt.plot(rt["N"], rt["Time_x86_mean"], "o-", label="x86 Time (mean)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Problem size N = nx·ny·nz (log)")
    plt.ylabel("Time (log)")
    plt.title("ARM vs x86: Runtime vs Problem Size (if comparable)")
    plt.legend()
    plt.show()

# --------------------------------
# 6) Export paper tables (optional)
# --------------------------------
arm_ps.to_csv(f"{BASE_DIR}/arm_per_size_table.csv", index=False)
x86_ps.to_csv(f"{BASE_DIR}/x86_per_size_table.csv", index=False)
merged.to_csv(f"{BASE_DIR}/arm_vs_x86_per_size_merged.csv", index=False)

print("Saved tables in:", BASE_DIR)
