import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# Nonlinear fit (PASCO-style)
from scipy.optimize import curve_fit

# -----------------------------
# PATHS (save everything next to this .py file)
# -----------------------------
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

iso_path = os.path.join(SCRIPT_DIR, "data - isoterm test2.txt")
adi_path = os.path.join(SCRIPT_DIR, "data - adibatic exp.txt")

# Cylinder radius (m) -> V = A*x
radius = 0.07
area = np.pi * radius**2

# -----------------------------
# LOAD + CLEAN
# -----------------------------
def load_pascalike(path):
    df = pd.read_csv(path, sep="\t")

    # Replace decimal commas with dots, then cast to float
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x) # type: ignore
    df = df.astype(float)

    # Standardize columns
    if df.shape[1] == 3:
        df.columns = ["Pressure_kPa", "Temperature_C", "Position_m"]
        df["Time"] = np.arange(len(df), dtype=float)
    elif df.shape[1] == 4:
        df.columns = ["Time", "Pressure_kPa", "Temperature_C", "Position_m"]
    else:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]}")

    # Derived quantities
    df["Temperature_K"] = df["Temperature_C"] + 273.15
    df["Volume_m3"] = df["Position_m"] * area  # approximate volume (no dead volume included)

    return df

iso = load_pascalike(iso_path)
adi = load_pascalike(adi_path)

# Save cleaned data
iso.to_csv(os.path.join(SCRIPT_DIR, "isoterm_cleaned.csv"), index=False)
adi.to_csv(os.path.join(SCRIPT_DIR, "adiabat_cleaned.csv"), index=False)

# -----------------------------
# METHOD 1: ln-ln (good for isothermal)
#   PV^n = const  -> ln P = -n ln V + C
# -----------------------------
def compute_n_loglog(df, v_col="Volume_m3", p_col="Pressure_kPa"):
    # Remove invalid points
    mask = (df[v_col] > 0) & (df[p_col] > 0)
    lnV = np.log(df.loc[mask, v_col].to_numpy())
    lnP = np.log(df.loc[mask, p_col].to_numpy())

    # Linear fit: lnP = a*lnV + b  -> n = -a
    a, b = np.polyfit(lnV, lnP, 1)
    n = -a
    return n, (a, b), lnV, lnP

n_iso_log, fit_iso_log, lnV_iso, lnP_iso = compute_n_loglog(iso)

# -----------------------------
# METHOD 2: PASCO-style nonlinear fit (recommended for adiabatic/polytropic)
#   P = A/(V - V0)^n + B
# -----------------------------
def pasco_inverse_power(x, A, x0, n, B):
    return A / np.power((x - x0), n) + B

def fit_adiabatic_pascolike(df, x_col="Position_m", p_col="Pressure_kPa",
                           x_min=0.01, x_max=None, fix_B_zero=True):
    x = df[x_col].to_numpy()
    P = df[p_col].to_numpy()

    # --- Choose the same type of interval as PASCO (ignore tiny-x cluster) ---
    if x_max is None:
        x_max = np.max(x)

    mask = (x > x_min) & (x < x_max) & np.isfinite(x) & np.isfinite(P)
    x_fit = x[mask]
    P_fit = P[mask]

    # Sort by x (helps stability)
    order = np.argsort(x_fit)
    x_fit, P_fit = x_fit[order], P_fit[order]

    # ----- Initial guesses (close to PASCO behavior) -----
    B0 = 0.0
    x0_0 = -0.05          # PASCO had negative x0
    n0  = 1.3
    A0  = (np.max(P_fit) - B0) * (np.min(x_fit) - x0_0)**n0

    if fix_B_zero:
        # Fit with B fixed to 0 (matches your screenshot B = 0.00)
        def model_fixedB(x, A, x0, n):
            return pasco_inverse_power(x, A, x0, n, 0.0)

        # Bounds: allow negative x0
        bounds_lower = [0.0, -1.0, 0.5]
        bounds_upper = [np.inf, np.min(x_fit) - 1e-6, 2.0]

        popt, pcov = curve_fit(
            model_fixedB, x_fit, P_fit,
            p0=[A0, x0_0, n0],
            bounds=(bounds_lower, bounds_upper),
            maxfev=300000
        )
        A, x0, n = popt
        perr = np.sqrt(np.diag(pcov))
        B = 0.0
        return (A, x0, n, B), (perr[0], perr[1], perr[2], 0.0), x_fit, P_fit

    else:
        # Fit with B free
        bounds_lower = [0.0, -1.0, 0.5, 0.0]
        bounds_upper = [np.inf, np.min(x_fit) - 1e-6, 2.0, 2000.0]

        popt, pcov = curve_fit(
            pasco_inverse_power, x_fit, P_fit,
            p0=[A0, x0_0, n0, B0],
            bounds=(bounds_lower, bounds_upper),
            maxfev=300000
        )
        perr = np.sqrt(np.diag(pcov))
        return popt, perr, x_fit, P_fit

(popt, perr, x_fit, P_fit) = fit_adiabatic_pascolike(adi, x_min=0.01, fix_B_zero=True)
A, x0, n_adi, B = popt
# -----------------------------
# PRINT RESULTS (for seminar slides)
# -----------------------------
print("\n=== Results for seminar ===")
print(f"Isothermal (ln-ln fit): n = {n_iso_log:.2f}  (near-perfect isothermal ~ 1.00)")


print(f"PASCO-like adiabatic fit: n = {n_adi:.2f}, x0 = {x0:.4f} m, B = {B:.2f} kPa")
# -----------------------------
# PLOTS FOR SEMINAR
# -----------------------------

# 1) PV plot (raw)
plt.figure()
plt.plot(iso["Volume_m3"], iso["Pressure_kPa"], label="Isoterm (långsam)")
plt.plot(adi["Volume_m3"], adi["Pressure_kPa"], label="Adiabat (snabb)")
plt.xlabel("Volym V (m^3) (≈ A·x)")
plt.ylabel("Tryck P (kPa)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "PV_diagram.png"))
plt.close()

# 2) TV plot
plt.figure()
plt.plot(iso["Volume_m3"], iso["Temperature_K"], label="Isoterm (långsam)")
plt.plot(adi["Volume_m3"], adi["Temperature_K"], label="Adiabat (snabb)")
plt.xlabel("Volym V (m^3) (≈ A·x)")
plt.ylabel("Temperatur T (K)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "TV_diagram.png"))
plt.close()

# 3) lnP-lnV for isothermal
plt.figure()
plt.scatter(lnV_iso, lnP_iso, s=12, label="Data")
a, b = fit_iso_log
plt.plot(lnV_iso, a * lnV_iso + b, label=f"Linjär fit (n={n_iso_log:.2f})")
plt.xlabel("ln(V)")
plt.ylabel("ln(P)")
plt.title("Isoterm: lnP vs lnV")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "lnP_lnV_isoterm.png"))
plt.close()

# -----------------------------
# PASCO-style fit plot (ADIABATIC)
# P = A / (x - x0)^n + B
# -----------------------------
plt.figure()

# Raw data used in the fit (same interval as PASCO)
plt.scatter(
    x_fit,
    P_fit,
    s=15,
    color="purple",
    label="Data (fit interval)"
)

# Smooth model curve
x_line = np.linspace(np.min(x_fit), np.max(x_fit), 600)
P_line = pasco_inverse_power(x_line, A, x0, n_adi, B)

plt.plot(
    x_line,
    P_line,
    color="gold",
    linewidth=2,
    label=f"PASCO-fit: n = {n_adi:.2f}"
)

plt.xlabel("Position x (m)")
plt.ylabel("Tryck P (kPa)")
plt.title("Adiabatisk/polytrop kompression (PASCO-stil)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(SCRIPT_DIR, "pasco_fit_adiabat.png"))
plt.close()

print("\nSaved figures in:", SCRIPT_DIR)
print("PV_diagram.png, TV_diagram.png, lnP_lnV_isoterm.png, pasco_fit_adiabat.png")
print("Cleaned data: isoterm_cleaned.csv, adiabat_cleaned.csv")
