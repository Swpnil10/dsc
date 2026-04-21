import pandas as pd

def load_data(file_name, label_mapping):
    print(f"Processing {file_name}...")
    df = pd.read_csv(file_name)
    df['text_combined'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df = df[['text_combined', 'label']]
    df['label'] = df['label'].map(label_mapping)
    return df.dropna()

spam_mapping = {0: 0, 1: 1}
phish_mapping = {0: 0, 1: 2}

print("--- EXTRACTING AND RELABELING ---")
enron = load_data('Enron.csv', spam_mapping)
ling = load_data('Ling.csv', spam_mapping)
ceas = load_data('CEAS_08.csv', spam_mapping)
spamassasin = load_data('SpamAssasin.csv', spam_mapping)
nazario = load_data('Nazario.csv', phish_mapping)
nigerian = load_data('Nigerian_Fraud.csv', phish_mapping)

# 1. Smash ALL data together without deleting anything!
master_df = pd.concat([enron, ling, ceas, spamassasin, nazario, nigerian], ignore_index=True)

# 2. Shuffle the rows so they aren't perfectly in order
final_df = master_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n--- FINAL DATASET READY ---")
print(f"Total Rows: {final_df.shape[0]}")
print(final_df['label'].value_counts())

final_df.to_csv('multiclass_emails.csv', index=False)
print("\nSaved successfully as 'multiclass_emails.csv'!")