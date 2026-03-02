import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ==================================
# Sample Dataset (Replace with real dataset in production)
# ==================================
data = {
    "text": [
        "Congratulations you won a lottery click here",
        "Urgent your account will be suspended verify now",
        "Meeting scheduled for tomorrow",
        "Please find attached project report",
        "Update your bank details immediately",
        "Team lunch at 1 PM",
        "Free gift card claim now",
        "Project discussion notes attached"
    ],
    "label": [1, 1, 0, 0, 1, 0, 1, 0]
}

# 1 = Phishing
# 0 = Legitimate

df = pd.DataFrame(data)

# ==================================
# Train/Test Split
# ==================================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.25,
    random_state=42,
    stratify=df["label"]
)

# ==================================
# Vectorization
# ==================================
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==================================
# Model Training
# ==================================
model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000
)

model.fit(X_train_vec, y_train)

# ==================================
# Evaluation
# ==================================
y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==================================
# Save Model & Vectorizer
# ==================================
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained and saved successfully.")
