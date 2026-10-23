# STOP PRESS - Fake News Detection

Large training datasets in `data/` are stored with Git LFS. Install Git LFS before cloning so those files are downloaded.

## Project structure

```
fake-news-detector/
├── app.py                  # Flask API + UI route
├── train_model.py          # leak-aware preprocessing, TF-IDF, training/eval
├── requirements.txt
├── data/                   #contains the WELFake Dataset csv file
├── model/                  
├── templates/
│   └── index.html          
└── static/
    ├── style.css
    └── script.js
```

## Next steps / stretch goals

- **Cross-source validation**: the real fix for the style-leakage issue
  above — hold out an entire source (not a random row split) so the test
  set forces the model to generalize across writing styles, not just
  memorize which corpus is which.
- **Bigger, cleaner test set**: build a small hand-labeled set of genuinely
  fresh real/fake examples (like the 6-sentence check in this README) and
  track accuracy on that as the real metric, separate from in-corpus
  accuracy.
- **Transformer upgrade**: fine-tune DistilBERT and compare — a good
  "before/after" story, especially paired with the cross-source test above.
- **Explainability**: add LIME or SHAP so a prediction shows which words/
  phrases drove the verdict, not just a label.
- **Deployment**: containerize with Docker and deploy (Render/Railway/Fly.io)
  so the demo is publicly link-able, same as SpamScope.
