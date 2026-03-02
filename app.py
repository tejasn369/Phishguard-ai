import os
from datetime import datetime
import re
from difflib import SequenceMatcher

from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import joblib
from markupsafe import escape

app = Flask(__name__)

# ==============================
# Database Configuration
# ==============================
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:password@localhost:5432/phishguard"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ==============================
# Database Model
# ==============================
class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    risk = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(50), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

with app.app_context():
    db.create_all()

# ==============================
# Load ML Model
# ==============================
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
except Exception as e:
    print("Model load error:", e)
    model = None
    vectorizer = None

# ==============================
# Configuration
# ==============================
TRUSTED_DOMAINS = ["google.com", "microsoft.com", "amazon.com"]
SUSPICIOUS_WORDS = ["urgent", "verify", "click here", "free", "update", "bank"]

# ==============================
# Core Functions
# ==============================
def calculate_risk(text):
    if not model or not vectorizer:
        return 0
    X = vectorizer.transform([text])
    probability = model.predict_proba(X)[0][1]
    return round(probability * 100, 2)


def highlight_keywords(text):
    safe_text = escape(text)
    for word in SUSPICIOUS_WORDS:
        safe_text = re.sub(
            word,
            f"<span style='color:red'>{word}</span>",
            safe_text,
            flags=re.IGNORECASE
        )
    return safe_text


def check_url_similarity(text):
    urls = re.findall(r'(https?://[^\s]+)', text)
    flagged = []
    for url in urls:
        domain = url.split("//")[-1].split("/")[0]
        for trusted in TRUSTED_DOMAINS:
            similarity = SequenceMatcher(None, domain, trusted).ratio()
            if similarity > 0.7 and domain != trusted:
                flagged.append(f"{domain} looks like {trusted}")
    return flagged

# ==============================
# Routes
# ==============================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.json.get("text", "")
    risk = calculate_risk(text)
    status = "High Risk" if risk > 70 else "Suspicious" if risk > 40 else "Safe"

    record = Analysis(content=text, risk=risk, status=status)
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "risk": risk,
        "status": status,
        "highlighted_text": highlight_keywords(text),
        "url_flags": check_url_similarity(text)
    })


@app.route("/dashboard")
def dashboard():
    records = Analysis.query.order_by(Analysis.created_at.desc()).all()
    total = len(records)

    if total > 0:
        avg = round(sum(r.risk for r in records) / total, 2)
        high = len([r for r in records if r.risk > 70])
        medium = len([r for r in records if 40 < r.risk <= 70])
        low = len([r for r in records if r.risk <= 40])
    else:
        avg = high = medium = low = 0

    return render_template(
        "dashboard.html",
        total=total,
        avg=avg,
        high=high,
        medium=medium,
        low=low
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
