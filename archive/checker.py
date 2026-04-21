import pandas as pd
import os

# List of the datasets we want to check
files = [
    'Enron.csv', 
    'SpamAssasin.csv', 
    'Nazario.csv', 
    'Nigerian_Fraud.csv', 
    'Ling.csv', 
    'CEAS_08.csv'
]

print("Checking column names...\n" + "-"*30)

for file in files:
    # Check if the file actually exists to avoid crashing
    if os.path.exists(file):
        # We use nrows=0 to just read the header (super fast) without loading the whole heavy file
        df = pd.read_csv(file, nrows=0) 
        print(f"✅ {file} columns: {list(df.columns)}")
    else:
        print(f"❌ Could not find {file} (Check the spelling or capitalization!)")
        