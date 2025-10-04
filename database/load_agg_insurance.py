# simple_csv_to_mysql.py
# -----------------------------------------
# Easiest way to load ONE CSV into MySQL.
# - Overwrites (replaces) the table each run to keep things simple.
# - Requires: pip install pandas SQLAlchemy mysql-connector-python
#
# How to run (example):
#   python simple_csv_to_mysql.py
#
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# === 1) EDIT THESE ===
CSV_PATH   = "/Users/kishore_kumar/PhonePe/pulse/csv_out/aggregated_insurance.csv"  # <- put your CSV path here
DB_USER    = "root"
DB_PASS    = "Root@123"        # special chars handled
DB_HOST    = "localhost"
DB_PORT    = 3306
DB_NAME    = "phonepe_pulse"   # database must exist (or see Step 3b below)
TABLE_NAME = "aggregated_insurance"       # name you want in MySQL

# === 2) Read the CSV ===
print(f"[INFO] Reading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"[INFO] Rows: {len(df)}  Columns: {list(df.columns)}")

# === 3) Connect to MySQL ===

pwd = quote_plus(DB_PASS)
engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{pwd}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

print(f"[INFO] Loading into MySQL table: {TABLE_NAME}")
df.to_sql(TABLE_NAME, engine, if_exists="replace", index=False, method="multi", chunksize=2000)
print("[SUCCESS] Done!")


