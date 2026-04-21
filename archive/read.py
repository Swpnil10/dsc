import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load and clean the new multiclass dataset
print("Loading the new multiclass dataset...")
df = pd.read_csv('multiclass_emails.csv')
df.dropna(inplace=True)

# 2. Separate text (X) and answers (y)
X = df['text_combined']
y = df['label']

# 3. Split into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Convert text to numbers using TF-IDF
print("Converting words to numbers...")
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Initialize and train the Logistic Regression model
print("Training the 3-category model... ")
# --- NEW: Added class_weight='balanced' ---
model = LogisticRegression(max_iter=1000, class_weight='balanced') 
model.fit(X_train_vectorized, y_train)

# 6. Test the model on the hidden 20% of data
print("Making predictions on the test data...")
predictions = model.predict(X_test_vectorized)

# 7. Evaluate how well it did
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Performance Report:")
# --- NEW: Added the 3rd category name here ---
print(classification_report(y_test, predictions, target_names=['Legit (0)', 'Spam (1)', 'Phishing/Scam (2)']))

# 8. Save the trained model and the vectorizer to your disk
print("\nSaving the upgraded model and vectorizer...")
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Saved successfully! Ready for the UI.")