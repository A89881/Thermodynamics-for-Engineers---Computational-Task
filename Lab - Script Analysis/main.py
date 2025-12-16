import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# FILE PATHS
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
iso_path = os.path.join(script_dir, "data - isoterm test2.txt")
adi_path = os.path.join(script_dir, "data - adibatic exp.txt")
radius = 0.07  # meters

# -----------------------------
# LOAD + CLEAN DATA
# -----------------------------
def load_pascalike(path):
    df = pd.read_csv(path, sep="\t")
    df = df.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x) # type: ignore
    df = df.astype(float)

    # Standardize columns
    if df.shape[1] == 3:
        df.columns = ["Pressure_kPa", "Temperature_C", "Position_m"]
    elif df.shape[1] == 4:
        df.columns = ["Time", "Pressure_kPa", "Temperature_C", "Position_m"]

    # Add derived quantities
    df["Temperature_K"] = df["Temperature_C"] + 273.15
    df["Volume"] = df["Position_m"]*np.pi*(radius)**2  # volume proxy

    return df

iso = load_pascalike(iso_path)
adi = load_pascalike(adi_path)

# -----------------------------
# POLYTROPIC EXPONENT n
# -----------------------------
def compute_n(df):
    mask = df["Volume"] > 0
    lnP = np.log(df.loc[mask, "Pressure_kPa"])
    lnV = np.log(df.loc[mask, "Volume"])
    coeffs = np.polyfit(lnV, lnP, 1)
    n = -coeffs[0]
    return n, coeffs

n_iso, fit_iso = compute_n(iso)
n_adi, fit_adi = compute_n(adi)

print("Polytropic exponent results:")
print(f"Isothermal-like: n = {n_iso:.2f}")
print(f"Adiabatic-like:  n = {n_adi:.2f}")

# -----------------------------
# PV DIAGRAM
# -----------------------------
plt.figure()
plt.plot(iso["Volume"], iso["Pressure_kPa"], label="Isoterm (långsam)")
plt.plot(adi["Volume"], adi["Pressure_kPa"], label="Adiabat (snabb)")
plt.xlabel("Volym (∝ kolvposition)")
plt.ylabel("Tryck (kPa)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "PV_diagram.png"))
plt.close()

# -----------------------------
# TV DIAGRAM
# -----------------------------
plt.figure()
plt.plot(iso["Volume"], iso["Temperature_K"], label="Isoterm (långsam)")
plt.plot(adi["Volume"], adi["Temperature_K"], label="Adiabat (snabb)")
plt.xlabel("Volym (∝ kolvposition)")
plt.ylabel("Temperatur (K)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "TV_diagram.png"))
plt.close()

# -----------------------------
# lnP - lnV PLOTS
# -----------------------------
def ln_plot(df, coeffs, title, filename):
    lnV = np.log(df["Volume"])
    lnP = np.log(df["Pressure_kPa"])
    plt.figure()
    plt.scatter(lnV, lnP, s=12, label="Data")
    plt.plot(lnV, coeffs[0]*lnV + coeffs[1], label="Linjär fit")
    plt.xlabel("ln(V)")
    plt.ylabel("ln(P)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, filename))
    plt.close()

ln_plot(iso, fit_iso, f"Isoterm: n = {n_iso:.2f}", "lnP_lnV_isoterm.png")
ln_plot(adi, fit_adi, f"Adiabat: n = {n_adi:.2f}", "lnP_lnV_adiabat.png")
