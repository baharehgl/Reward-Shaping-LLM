import pandas as pd

# 1) Read the HDF file (where you know it works, e.g. base Anaconda environment)
df = pd.read_hdf("KPI_data/test/phase2_ground_truth.hdf")

# 2) If your environment code expects "anomaly" instead of "label", rename columns:
if "label" in df.columns and "anomaly" not in df.columns:
    df.rename(columns={"label": "anomaly"}, inplace=True)

# 3) If your environment only needs ['value', 'anomaly'] (and possibly 'timestamp'),
#    drop any extra columns that might confuse it:
if "KPI ID" in df.columns:
    df.drop(columns=["KPI ID"], inplace=True)

# 4) Save to CSV
df.to_csv("KPI_data/test/phase2_ground_truth.csv", index=False)
print("Converted phase2_ground_truth.hdf to phase2_ground_truth.csv!")
