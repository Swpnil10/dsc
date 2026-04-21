import joblib
import numpy as np

# 1. Load the objects from your archive folder
# Make sure the path matches where your files are actually stored!
model = joblib.load('archive/phishing_model.pkl')
vectorizer = joblib.load('archive/vectorizer.pkl')

# 2. Now the 'vectorizer' variable exists, so this will work:
words = vectorizer.get_feature_names_out()

# 3. Get the weights for the Phishing category (Label 2)
phish_weights = model.coef_[2]

# 4. Pair words with their weights and sort them
top_phish_words = sorted(zip(phish_weights, words), reverse=True)[:20]

print("Top 20 Words Indicating Phishing:")
for weight, word in top_phish_words:
    print(f"{word}: {weight:.4f}")