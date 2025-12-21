import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import pyfluids as pf


# This script scans the turbine extraction pressure in a Rankine cycle with an Open feedwater heater (FWH)
# to find the maximum thermal efficiency. It also compares against the same Rankine cycle WITHOUT FWH.
# Outputs:
# - CSV with eta(P_extract) and y(P_extract)
# - Two plots: eta vs pressure, y vs pressure
# - A short summary.txt

# Auto-locate folder for saving outputs
base_dir = Path("Beräkningsuppgift-2")
base_dir.mkdir(parents=True, exist_ok=True)

# Create Fluid object representing water (pyfluids)
WATER = pf.Fluid(pf.FluidsList.Water)


# Helper functions (pyfluids)

# Determine enthalpy h and entropy s at a given pressure and temperature
def props_pT(P_pa: float, T_C: float):
    WATER.update(pf.Input.pressure(P_pa), pf.Input.temperature(T_C))
    return WATER.enthalpy, WATER.entropy  # J/kg, J/kgK


# Determine enthalpy h and entropy s at a given pressure and vapour quality x (x=0 => saturated liquid, x=1 => saturated vapour)
def props_px(P_pa: float, x: float):
    WATER.update(pf.Input.pressure(P_pa), pf.Input.quality(x))
    return WATER.enthalpy, WATER.entropy

# Determine enthalpy h at (P, s). Used for isentropic turbine steps (s_out = s_in)
def h_ps(P_pa: float, s_J_per_kgK: float):
    WATER.update(pf.Input.pressure(P_pa), pf.Input.entropy(s_J_per_kgK))
    return WATER.enthalpy

# Determine saturated liquid specific volume v_f at pressure P (needed for pump work ~ v*Δp)
def v_f_sat(P_pa: float):
    WATER.update(pf.Input.pressure(P_pa), pf.Input.quality(0.0))
    return WATER.specific_volume

# Determine condenser temperature fixes condenser pressure (saturation pressure at Tcond)
def condenser_pressure_from_Tcond(Tcond_C: float) -> float:
    WATER.update(pf.Input.temperature(Tcond_C), pf.Input.quality(0.0))
    return WATER.pressure  # Pa

# Determine pump work per kg for (almost) incompressible liquid
def pump_work_vdp(P_in: float, P_out: float) -> float:
    # w_p ≈ v_f(P_in) * (P_out - P_in)
    v = v_f_sat(P_in)
    return v * (P_out - P_in)  # J/kg


# Core cycle formulas and functions

# Computes extraction fraction y in an OPEN feedwater heater (direct mixing)
def calc_y_open_fwh(h6: float, h2: float, h3: float) -> float:
    # Condition given in assignment: after the FWH we want saturated liquid (state 3)
    # Energy balance: y*h6 + (1-y)*h2 = h3  =>  y = (h3 - h2)/(h6 - h2)
    return (h3 - h2) / (h6 - h2)


def efficiency_open_fwh(P_extract: float,
                        P_boiler: float = 15e6,
                        T_max_C: float = 600.0,
                        T_cond_C: float = 30.0):
    # Purpose: compute thermal efficiency eta for ONE chosen extraction pressure (this is what we scan)
    #
    # State convention (typical for Rankine + open FWH):
    # 1: condenser outlet, sat. liquid at P_cond
    # 2: after Pump I to P_extract
    # 3: after open FWH, sat. liquid at P_extract
    # 4: after Pump II to P_boiler
    # 5: boiler outlet / turbine inlet at (P_boiler, T_max)
    # 6: turbine state at P_extract (bleed stream)
    # 7: turbine outlet at P_cond for the remaining flow (1-y)

    # Condenser pressure from condenser temperature
    P_cond = condenser_pressure_from_Tcond(T_cond_C)

    # State 1: saturated liquid at condenser pressure
    h1, s1 = props_px(P_cond, 0.0)

    # Pump I: 1 -> 2
    w_p1 = pump_work_vdp(P_cond, P_extract)
    h2 = h1 + w_p1

    # State 3: saturated liquid at extraction pressure (required by problem statement)
    h3, s3 = props_px(P_extract, 0.0)

    # Pump II: 3 -> 4
    w_p2 = pump_work_vdp(P_extract, P_boiler)
    h4 = h3 + w_p2

    # Boiler: 4 -> 5 (max T)
    h5, s5 = props_pT(P_boiler, T_max_C)

    # Turbine: isentropic expansions
    # 5 -> 6 to extraction pressure (bleed point)
    h6 = h_ps(P_extract, s5)

    # 5 -> 7 to condenser pressure (for remaining flow)
    h7 = h_ps(P_cond, s5)

    # Extraction fraction y from open FWH mixing requirement
    y = calc_y_open_fwh(h6, h2, h3)

    # If y is not physically valid, return NaN eta so scan can ignore it
    if (y < 0.0) or (y > 1.0) or np.isnan(y):
        return {"eta": np.nan, "y": y, "P_extract": P_extract, "P_cond": P_cond}

    # Turbine work per kg entering turbine:
    # first stage uses full flow, second stage only (1-y)
    W_turb = (h5 - h6) + (1.0 - y) * (h6 - h7)

    # Pump work per kg (same 1 kg basis in this setup)
    W_pumps = w_p1 + w_p2

    # Heat added in boiler
    Q_in = (h5 - h4)

    # Thermal efficiency
    eta = (W_turb - W_pumps) / Q_in

    return {"eta": eta, "y": y, "P_extract": P_extract, "P_cond": P_cond}


# Computes thermal efficiency of the same Rankine cycle but WITHOUT feedwater heating
# (no extraction, no open FWH, only one pump)
def efficiency_no_fwh(P_boiler: float = 15e6,
                      T_max_C: float = 600.0,
                      T_cond_C: float = 30.0):

    P_cond = condenser_pressure_from_Tcond(T_cond_C)

    # State 1: saturated liquid at condenser pressure
    h1, s1 = props_px(P_cond, 0.0)

    # Single pump: 1 -> 2 to boiler pressure
    w_p = pump_work_vdp(P_cond, P_boiler)
    h2 = h1 + w_p

    # Boiler: 2 -> 3
    h3, s3 = props_pT(P_boiler, T_max_C)

    # Turbine: 3 -> 4 isentropic to condenser pressure
    h4 = h_ps(P_cond, s3)

    # Work/heat
    W_turb = (h3 - h4)
    Q_in = (h3 - h2)
    eta = (W_turb - w_p) / Q_in

    return {"eta": eta, "P_cond": P_cond}

# Loops over many extraction pressures and store eta(P) and y(P)
def scan_pressures(P_min: float, P_max: float, n: int = 120,
                   P_boiler: float = 15e6, T_max_C: float = 600.0, T_cond_C: float = 30.0):
    P_list = np.linspace(P_min, P_max, n)

    eta_list = []
    y_list = []

    for P_ex in P_list:
        res = efficiency_open_fwh(P_extract=float(P_ex),
                                  P_boiler=P_boiler, T_max_C=T_max_C, T_cond_C=T_cond_C)
        eta_list.append(res["eta"])
        y_list.append(res["y"])

    return P_list, np.array(eta_list), np.array(y_list)


# Given limits from assignment
P_boiler = 15e6      # Pa (15 MPa)
T_max_C = 600.0      # °C
T_cond_C = 30.0      # °C

# Condenser pressure from saturation at 30°C
P_cond = condenser_pressure_from_Tcond(T_cond_C)
print("Data ready (water properties via pyfluids)")
print(f"Condenser pressure from T_cond = {T_cond_C}°C: {P_cond/1e5:.3f} bar")

# We scan extraction pressure between condenser pressure and boiler pressure
P_min = P_cond * 1.05
P_max = P_boiler * 0.95

P_list, etas, ys = scan_pressures(P_min, P_max, n=140,
                                  P_boiler=P_boiler, T_max_C=T_max_C, T_cond_C=T_cond_C)

# Find maximum efficiency (ignore NaNs)
idx = np.nanargmax(etas)
P_opt = P_list[idx]
eta_opt = etas[idx]
y_opt = ys[idx]

print("\nQuestion 1) Open FWH optimisation")
print(f"Optimal extraction pressure: {P_opt/1e6:.4f} MPa")
print(f"Maximum thermal efficiency: {eta_opt:.6f}")
print(f"Extraction fraction y at optimum: {y_opt:.6f}")

# Save scan results to CSV (so you can reuse numbers without re-running scan)
df_scan = pd.DataFrame({
    "P_extract_Pa": P_list,
    "P_extract_MPa": P_list / 1e6,
    "eta": etas,
    "y": ys
})
scan_csv = base_dir / "q1_eta_vs_extraction_pressure.csv"
df_scan.to_csv(scan_csv, index=False)
print(f"Scan results saved to {scan_csv}")

# Plot eta vs extraction pressure 
plot_kw = dict(figsize=(12, 5), linewidth=0.9)
save_kw = dict(dpi=200, bbox_inches='tight')

plt.figure(**plot_kw)  # type: ignore
plt.plot(P_list/1e6, etas)
plt.title("Thermal efficiency vs extraction pressure (Open FWH)")
plt.xlabel("Extraction pressure (MPa)")
plt.ylabel("Thermal efficiency (-)")
plt.grid(True)
plt.tight_layout()
plt.savefig(base_dir / "q1_eta_vs_pressure.png", **save_kw)

# Plot y vs extraction pressure
plt.figure(**plot_kw)  # type: ignore
plt.plot(P_list/1e6, ys)
plt.title("Extraction fraction y vs extraction pressure (Open FWH)")
plt.xlabel("Extraction pressure (MPa)")
plt.ylabel("y (-)")
plt.grid(True)
plt.tight_layout()
plt.savefig(base_dir / "q1_y_vs_pressure.png", **save_kw)

plt.show()
print("Figures saved:",
      base_dir / "q1_eta_vs_pressure.png",
      base_dir / "q1_y_vs_pressure.png")

# Question 2: compare with cycle without feedwater heating
res_no = efficiency_no_fwh(P_boiler=P_boiler, T_max_C=T_max_C, T_cond_C=T_cond_C)
eta_no = res_no["eta"]

print("\nQuestion 2) Comparison (no feedwater heating)")
print(f"Thermal efficiency without FWH: {eta_no:.6f}")
print(f"Thermal efficiency with optimal Open FWH: {eta_opt:.6f}")
print(f"Difference (eta_opt - eta_no): {(eta_opt - eta_no):.6f}")

# Save a short summary file with key results
summary_txt = base_dir / "summary.txt"
with open(summary_txt, "w", encoding="utf-8") as f:
    f.write("Beräkningsuppgift 2: Rankinecykel med öppen matarvattenförvärmning\n")
    f.write(f"T_cond = {T_cond_C} °C -> P_cond = {P_cond/1e5:.4f} bar\n")
    f.write(f"P_boiler = {P_boiler/1e6:.3f} MPa, T_max = {T_max_C} °C\n\n")
    f.write("Open FWH optimum:\n")
    f.write(f"  P_extract* = {P_opt/1e6:.6f} MPa\n")
    f.write(f"  eta_max    = {eta_opt:.8f}\n")
    f.write(f"  y*         = {y_opt:.8f}\n\n")
    f.write("No FWH:\n")
    f.write(f"  eta_noFWH  = {eta_no:.8f}\n")
    f.write(f"  delta eta  = {(eta_opt - eta_no):.8f}\n")

print(f"Summary saved to {summary_txt}")
