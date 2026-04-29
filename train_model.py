import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# ── Step 1: Load cleaned data ──────────────────────────
print("Loading cleaned data...")
df = pd.read_csv("cleaned_data.csv")
df = df.dropna(subset=["content"])

# ── Step 2: Split into X (input) and y (label) ─────────
X = df["content"]
y = df["label"]

# ── Step 3: Split into train and test sets ─────────────
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── Step 4: TF-IDF Vectorizer ──────────────────────────
# Converts text into numbers the model can understand
print("\nApplying TF-IDF...")
tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=50000
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)
print("TF-IDF done!")

# ── Step 5: Train the model ────────────────────────────
print("\nTraining model...")
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)
print("Training done!")

# ── Step 6: Test the model ─────────────────────────────
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# ── Step 7: Confusion Matrix ───────────────────────────
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  True Fake  (correctly detected fake): {cm[0][0]}")
print(f"  True Real  (correctly detected real): {cm[1][1]}")
print(f"  Wrong Fake (real marked as fake):     {cm[0][1]}")
print(f"  Wrong Real (fake marked as real):     {cm[1][0]}")

# ── Step 8: Save model and vectorizer ──────────────────
joblib.dump(model, "model.pkl")
joblib.dump(tfidf,  "tfidf.pkl")
print("\nModel saved as model.pkl")
print("Vectorizer saved as tfidf.pkl")
print("\nPhase 3 Complete!")