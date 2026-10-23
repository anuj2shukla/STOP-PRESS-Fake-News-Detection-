import argparse
import os
import re
import string
import sys
import joblib
import pandas as pd
import nltk
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("english"))

LEAK_WORDS = {"reuters", "afp"}

SOURCE_TAG_PATTERNS = [
    re.compile(r"\(reuters\)", re.IGNORECASE),
    re.compile(r"\(ap\)", re.IGNORECASE),
    re.compile(r"\(afp\)", re.IGNORECASE),

    re.compile(r"^[A-Z][A-Za-z.,\s]{0,40}\([^)]{1,15}\)\s*-\s*"),
]

def strip_source_tags(text: str) -> str:
    for pattern in SOURCE_TAG_PATTERNS:
        text = pattern.sub(" ", text)
    return text

def clean_text(text: str, strip_leaks: bool = True) -> str:
    text = str(text)
    if strip_leaks:
        text = strip_source_tags(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = [w for w in text.split() if w not in STOPWORDS and w not in LEAK_WORDS and len(w) > 2]
    return " ".join(tokens)


def load_welfake_dataset(path: str) -> pd.DataFrame:
    
    df = pd.read_csv(path)
    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")
    df["text"] = (df["title"] + " " + df["text"]).str.strip()
    df["label"] = df["label"].map({0: "REAL", 1: "FAKE"})
    df = df[["text", "label"]].dropna()
    df = df[df["text"].str.len() > 0]
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_kaggle_dataset(fake_path: str, true_path: str) -> pd.DataFrame:
    fake = pd.read_csv(fake_path)
    true = pd.read_csv(true_path)
    fake["label"] = "FAKE"
    true["label"] = "REAL"
    df = pd.concat([fake, true], ignore_index=True)
    if "title" in df.columns and "text" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    df = df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_feedback(path: str):
    """Load human-verified examples saved by the app (data/feedback.csv)."""
    if not os.path.exists(path):
        return None
    fb = pd.read_csv(path)
    fb = fb[["text", "label"]].dropna()
    fb = fb[fb["label"].isin(["REAL", "FAKE"])]
    fb = fb.drop_duplicates(subset="text").reset_index(drop=True)
    return fb if len(fb) else None


def train_and_eval(df: pd.DataFrame, strip_leaks: bool, max_features: int = None,
                   extra: pd.DataFrame = None, extra_weight: int = 20):
    df = df.copy()
    df["clean_text"] = df["text"].apply(lambda t: clean_text(t, strip_leaks=strip_leaks))
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    if extra is not None and len(extra):
        extra_clean = extra["text"].apply(lambda t: clean_text(t, strip_leaks=strip_leaks))
        X_train = pd.concat([X_train] + [extra_clean] * extra_weight, ignore_index=True)
        y_train = pd.concat([y_train] + [extra["label"]] * extra_weight, ignore_index=True)

    vectorizer = TfidfVectorizer(max_df=0.7, min_df=2, ngram_range=(1, 2), max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)  
    X_test_tfidf = vectorizer.transform(X_test)         

    candidates = {
        
        "PassiveAggressive (SGD)": SGDClassifier(
            loss="hinge", penalty=None, learning_rate="pa1", eta0=1.0,
            max_iter=100, random_state=42,
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000),
    }

    best_name, best_model, best_acc = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)
        acc = accuracy_score(y_test, preds)
        tag = "leak-stripped" if strip_leaks else "LEAKY (source tags kept)"
        print(f"\n--- {name} [{tag}] ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, zero_division=0))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))
        if acc > best_acc:
            best_name, best_model, best_acc = name, model, acc

    return best_name, best_model, vectorizer, best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--welfake", type=str, default="data/WELFake_Dataset.csv",
                         help="Path to WELFake_Dataset.csv (default: data/WELFake_Dataset.csv)")
    parser.add_argument("--fake", type=str, default="data/Fake.csv",
                         help="Path to Kaggle Fake.csv (default: data/Fake.csv)")
    parser.add_argument("--true", type=str, default="data/True.csv",
                         help="Path to Kaggle True.csv (default: data/True.csv)")
    parser.add_argument("--max-features", type=int, default=60000,
                         help="Cap TF-IDF vocabulary size (keeps large datasets fast/memory-safe)")
    parser.add_argument("--skip-leaky-baseline", action="store_true",
                         help="Skip training the leaky (unstripped) comparison model")
    parser.add_argument("--feedback", type=str, default="data/feedback.csv",
                         help="CSV of human-verified examples collected from the app")
    parser.add_argument("--feedback-weight", type=int, default=20,
                         help="How many times each feedback row is repeated in training")
    args = parser.parse_args()

    if args.fake and args.true:
        print(f"Loading Kaggle dataset from {args.fake} / {args.true} ...")
        df = load_kaggle_dataset(args.fake, args.true)
    elif os.path.exists(args.welfake):
        print(f"Loading WELFake dataset from {args.welfake} ...")
        df = load_welfake_dataset(args.welfake)
    else:
        print(f"ERROR: no dataset found at '{args.welfake}'.")
        print("Place WELFake_Dataset.csv in data/, or pass --welfake <path>.")
        print("(Not falling back to a toy dataset — a tiny fallback set is exactly")
        print(" what produces misleading numbers like a flat 50% accuracy.)")
        sys.exit(1)

    print(f"Dataset size: {len(df)} rows | label counts:\n{df['label'].value_counts()}")

    feedback = load_feedback(args.feedback)
    fb_train, fb_test = None, None
    if feedback is not None:
        feedback = feedback.sample(frac=1, random_state=42).reset_index(drop=True)
        n_test = int(len(feedback) * 0.3) if len(feedback) >= 10 else 0
        fb_test = feedback.iloc[:n_test] if n_test else None
        fb_train = feedback.iloc[n_test:]
        print(f"Feedback rows: {len(feedback)} (train: {len(fb_train)}, held-out test: {n_test})")
    else:
        print(f"No feedback file found at '{args.feedback}' — training on the base dataset only.")

    if not args.skip_leaky_baseline:
        
        print(" BASELINE — source tags left in (this is the misleading number)")
        train_and_eval(df, strip_leaks=False, max_features=args.max_features)

    
    print(" PRODUCTION MODEL — source tags stripped (the honest number)")
    best_name, best_model, vectorizer, best_acc = train_and_eval(
        df, strip_leaks=True, max_features=args.max_features,
        extra=fb_train, extra_weight=args.feedback_weight,
    )

    print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

    if fb_test is not None:
        X_fb = vectorizer.transform(fb_test["text"].apply(clean_text))
        fb_acc = accuracy_score(fb_test["label"], best_model.predict(X_fb))
        print(f"Real-world accuracy on {len(fb_test)} held-out feedback examples: {fb_acc:.4f}")

    joblib.dump(best_model, "model/model.pkl")
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    print("Saved model/model.pkl and model/vectorizer.pkl")


if __name__ == "__main__":
    main()