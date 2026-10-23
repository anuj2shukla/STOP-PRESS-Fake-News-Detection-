
import os
import math
import joblib
from flask import Flask, request, jsonify, render_template 
import csv
import threading
from datetime import datetime, timezone   

from train_model import clean_text

app = Flask(__name__)
DATA_DIR = "data"
PRED_LOG = os.path.join(DATA_DIR, "predictions_log.csv")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback.csv")
file_lock = threading.Lock()


def append_row(path, header, row):
    os.makedirs(DATA_DIR, exist_ok=True)
    with file_lock:
        is_new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(header)
            writer.writerow(row)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

MODEL_PATH = "model/model.pkl"
VECTORIZER_PATH = "model/vectorizer.pkl"

model = None
vectorizer = None
feature_names = None
coefs = None

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    if hasattr(model, "coef_"):
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
       

def get_reasons(vec, top_n=5):
    if coefs is None:
        return [], []
    

    row = vec.tocoo()
    contributions = [(feature_names[idx], float(val) * float(coefs[idx]))
                      for idx, val in zip(row.col, row.data)]
    contributions.sort(key=lambda x: x[1])

    fake_reasons = [w for w, c in contributions if c < 0][:top_n]
    real_reasons = [w for w, c in reversed(contributions) if c > 0][:top_n]
    return real_reasons, fake_reasons


def get_scores(vec):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(vec)[0]  
        fake_pct, real_pct = float(proba[0]) * 100, float(proba[1]) * 100
    elif hasattr(model, "decision_function"):
        score = float(model.decision_function(vec)[0])  
        real_pct = 1 / (1 + math.exp(-score)) * 100
        fake_pct = 100 - real_pct
    else:
        real_pct, fake_pct = 50.0, 50.0
    return round(real_pct, 1), round(fake_pct, 1)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not trained yet. Run train_model.py first."}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided."}), 400

    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    real_pct, fake_pct = get_scores(vec)
    real_reasons, fake_reasons = get_reasons(vec)

    append_row(
        PRED_LOG,
        ["timestamp", "text", "prediction", "real_pct", "fake_pct"],
        [now_iso(), text, prediction, real_pct, fake_pct],
    )

    return jsonify({
        "prediction": prediction,
        "confidence": real_pct if prediction == "REAL" else fake_pct,
        "scores": {"REAL": real_pct, "FAKE": fake_pct},
        "reasons": {"REAL": real_reasons, "FAKE": fake_reasons},
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()
    prediction = data.get("prediction", "")
    true_label = data.get("true_label", "")

    if not text or true_label not in ("REAL", "FAKE"):
        return jsonify({"error": "Invalid feedback."}), 400

    append_row(
        FEEDBACK_FILE,
        ["timestamp", "text", "model_prediction", "label"],
        [now_iso(), text, prediction, true_label],
    )
    return jsonify({"status": "saved"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
