# Stop Press: Fake News Detection

A web app that reads a news headline, paragraph or full article and classifies it as **REAL** or **FAKE**. It is a binary text classifier built with TF-IDF features and a linear model, served through Flask with a newspaper-desk style interface.

> Large training datasets in `data/` are stored with Git LFS. Install Git LFS before cloning so those files are downloaded.
> 
## Features

- REAL / FAKE verdict with a score bar showing the percentage for each class
- "Why" lists showing the words that pushed the decision toward REAL or FAKE
- Text input by typing, pasting, or uploading/pasting a screenshot (read in the browser with OCR)
- Every prediction is saved to `data/predictions_log.csv`
- Yes / No feedback buttons save human-verified labels to `data/feedback.csv`, which are added to the training data the next time the model is retrained

## How it works

1. **Preprocessing:** lowercase, remove links, punctuation, numbers and stopwords, and strip wire-service tags such as `(Reuters)` and `(AP)`.
2. **Feature extraction:** TF-IDF over single words and two-word phrases, with the vocabulary capped at 60,000 terms.
3. **Classification:** a Passive-Aggressive classifier and a Logistic Regression model are both trained, and the one with higher test accuracy is kept.
4. **Prediction:** the app returns the label, the REAL/FAKE scores, and the top words for each side.
5. **Explanation:** each word's TF-IDF weight is multiplied by the model's coefficient for that word. This shows what the model learned from the data, not whether a claim is true.

## Results

Trained on the WELFake dataset (72,134 articles).

| Measure | Result |
|---|---|
| Test accuracy, source tags left in | about 97.3% |
| Test accuracy, source tags removed | about 96.8% |
| Hand test on 6 freshly written sentences | 4 of 6 correct |

### Important limitation

The high test accuracy overstates real-world performance. WELFake's REAL articles come mostly from wire-service journalism and its FAKE articles from other kinds of sources, so the two classes differ in writing style and vocabulary, not only in truthfulness. A random train/test split keeps both parts in the same style, so the score shows the model can tell the two source groups apart. It does not show that the model can detect misinformation in general. Removing the source tags barely changed accuracy, which shows the effect comes from style and not from one removable string.

Treat the output as a word-pattern signal, not a fact-check.

## Setup

```bash
git lfs install
git clone <this-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

### Train the model

Make sure `data/WELFake_Dataset.csv` is present, then run:

```bash
python train_model.py
```

This prints accuracy with and without the source tags, then saves `model/model.pkl` and `model/vectorizer.pkl`. If `data/feedback.csv` exists, its verified examples are included in training.

### Run the app

```bash
python app.py
```

Open `http://localhost:5000`, paste an article, and click **Run verification**. Restart the app after retraining so it loads the new model.

## Project structure

```
STOP-PRESS-Fake-News-Detection-/
├── app.py                  # Flask API, prediction and feedback routes
├── train_model.py          # preprocessing, TF-IDF, training and evaluation
├── requirements.txt
├── data/                   # WELFake_Dataset.csv; predictions_log.csv and feedback.csv are created by the app
├── model/                  # model.pkl and vectorizer.pkl, created by training
├── templates/
│   └── index.html          # user interface
└── static/
    ├── style.css
    └── script.js
```

## Next steps

- **Cross-source validation:** hold out entire publishers for testing so the test measures generalisation across writing styles, not just memorised style.
- **Fresh hand-labelled test set:** track accuracy on new, human-checked examples as the main metric.
- **More human feedback:** collect many verified examples through the Yes / No buttons and retrain.
- **Deployment:** containerise with Docker and host the demo publicly.

## References

- Verma, P. K., Agrawal, P., Amorim, I., and Prodan, R. (2021). WELFake: Word Embedding Over Linguistic Features for Fake News Detection. *IEEE Transactions on Computational Social Systems*, 8(4), 881-893.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
