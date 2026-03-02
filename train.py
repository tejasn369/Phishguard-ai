import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

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
    "label": [1,1,0,0,1,0,1,0]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["text"])
y = df["label"]

model = LogisticRegression()
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model trained successfully.")