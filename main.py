import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load the temperature data from the text file
data = np.loadtxt(r"Uppsala_temperaturer_2008_2017-1.txt")
print(f"Data loaded successfully")

# Extract the relevant columns into csv for easier handling with pandas
year = data[:, 0]
day = data[:, 1]
temperatures = data[:, 2]
df = pd.DataFrame({'Year': year, "Day": day, 'Temperature': temperatures})
export_filename = 'Uppsala_temperaturer_2008_2017.csv'
df.to_csv(export_filename, index=False)
print(f"Data exported to {export_filename}")

# Read the CSV file using pandas
df = pd.read_csv(export_filename)

# Build a real datetime axis (much easier to read/scale plots)
# Assumes "Day" is day-of-year (1..365/366)
df['Date'] = pd.to_datetime(df['Year'].astype(int), format='%Y') + \
             pd.to_timedelta(df['Day'].astype(int) - 1, unit='D')

# Calculate the radiator temperature based on the given formula
def calc_T_rad(T_out_l:list, T_in: float):
    T_rad_l = []
    for T_out in T_out_l:
        if T_out < 0.0:
            T_rad_l.append(T_in*(1+0.043*T_in+0.035*abs(T_out)))
        elif 0.0 <= T_out < T_in:
            T_rad_l.append(T_in*(1+0.043*(T_in-T_out)))
        else:
            T_rad_l.append(0.0)  # No radiator needed for T_out >= T_in
    return [round(temp, 1) for temp in T_rad_l]


calc_T_rad_l = calc_T_rad(df['Temperature'], 21)  # type: ignore
df['Radiator_Temperature'] = calc_T_rad_l
print("Radiator temperatures calculated and added to DataFrame")

#Question 1.a) Calculate the heatloss (värmelackage) in (kWh/dag)
def calc_heatloss_kWh_day(T_out_l:list, T_in: float):
    heatloss_l = []
    for T_out in T_out_l:
        if T_out < T_in:
            Qdot_MJ_per_h = 2.0*(T_in - T_out)      # MJ/h
            Qday_MJ = Qdot_MJ_per_h*24.0           # MJ/day
            Qday_kWh = Qday_MJ/3.6                # kWh/day  (1 kWh = 3.6 MJ)
            heatloss_l.append(Qday_kWh)
        else:
            heatloss_l.append(0.0)
    return heatloss_l


df['Heatloss_kWh_day'] = calc_heatloss_kWh_day(df['Temperature'], 21)  # type: ignore
print("Heatloss calculated and added to DataFrame")

#Question 1.b) Carnot COP for each day
def calc_COP(T_rad_l:list, T_cold_C: float = 10.0):
    COP_l = []
    for T_rad in T_rad_l:
        if T_rad > 0.0:
            T_H = T_rad + 273.15
            T_C = T_cold_C + 273.15
            COP_l.append(T_H/(T_H - T_C))
        else:
            COP_l.append(0.0)
    return COP_l


df['COP'] = calc_COP(df['Radiator_Temperature'])  # type: ignore
print("COP calculated and added to DataFrame")

#Question 1.c) Electricity use of the heat pump per day in kWh/day
def calc_pump_electricity_kWh_day(heatloss_l:list, COP_l:list):
    el_l = []
    for Qloss, cop in zip(heatloss_l, COP_l):
        if cop > 0.0:
            el_l.append(Qloss/cop)
        else:
            el_l.append(0.0)
    return el_l


df['Pump_Electricity_kWh_day'] = calc_pump_electricity_kWh_day(
    df['Heatloss_kWh_day'], df['COP']  # type: ignore
)
print("Pump electricity calculated and added to DataFrame")

# Save everything back to CSV
df.to_csv('Uppsala_temperaturer_2008_2017.csv', index=False)
print("Final data saved to CSV")

# ---- Plots (Question 1) ----
# Make plots bigger + readable, and save them
plot_kw = dict(figsize=(12, 5), linewidth=0.8)
save_kw = dict(dpi=200, bbox_inches='tight')

# 1a Heat loss
plt.figure(**plot_kw) # type: ignore
plt.plot(df['Date'], df['Heatloss_kWh_day'])
plt.title("Heat loss (kWh/day) over time")
plt.xlabel("Date")
plt.ylabel("kWh/day")
plt.grid(True)
plt.tight_layout()
plt.savefig("q1a_heatloss.png", **save_kw)

# 1b COP
plt.figure(**plot_kw) # type: ignore
plt.plot(df['Date'], df['COP'])
plt.title("Carnot COP over time")
plt.xlabel("Date")
plt.ylabel("COP")
plt.grid(True)
plt.tight_layout()
plt.savefig("q1b_COP.png", **save_kw)

# 1c Electricity use
plt.figure(**plot_kw) # type: ignore
plt.plot(df['Date'], df['Pump_Electricity_kWh_day'])
plt.title("Heat pump electricity use (kWh/day) over time")
plt.xlabel("Date")
plt.ylabel("kWh/day")
plt.grid(True)
plt.tight_layout()
plt.savefig("q1c_pump_electricity.png", **save_kw)

plt.show()
print("Figures saved: q1a_heatloss.png, q1b_COP.png, q1c_pump_electricity.png")
