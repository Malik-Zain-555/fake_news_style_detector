# Fake or Real News: What Does the Model Actually Learn?

A TF-IDF + Logistic Regression classifier that separates fake from real news with about 98% accuracy, and an investigation showing that it mostly recognises **the writing style of the source**, not whether the news is true.

## Key finding

High accuracy did not mean the model understood truth. After checking the model's weights and testing it on my own texts, it turned out to rely on style: news-agency phrasing pushes toward "real", blog/clickbait phrasing pushes toward "fake".

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle), loaded with `kagglehub`.

| File | Articles |
|---|---|
| `Fake.csv` | 23,481 |
| `True.csv` | 21,417 |

Combined and shuffled: 44,898 articles. Labelled `fake_news` = `Yes` / `No`.

## Approach

1. Combine both files, add a `fake_news` label, shuffle.
2. Clean the text: lowercase; remove URLs and emails; keep letters only.
3. Split 80/20 (35,918 train / 8,980 test), stratified by label, `random_state=42`.
4. `TfidfVectorizer` fit on the training set only, then applied to the test set.
5. `LogisticRegression(max_iter=1000)`.
6. Cross-validation with a `Pipeline` (vectorizer inside) so each fold fits the vectorizer only on its own training part.

## Results

| Setup | Test accuracy | 5-fold CV (3,000 features) |
|---|---|---|
| Baseline | 98.8% | 98.7% |
| After removing the word "reuters" | 98.2% | 97.9% |

## Why the accuracy is misleading

1. **One word almost separates the classes.** In the raw text, "Reuters" appears in 21,378 of 21,417 real articles (99.8%) and in only 311 of 23,481 fake ones.
2. **Removing it barely changed accuracy** (about 0.6 points), so other style markers carry the same signal.
3. **Top model weights are style words**, not facts (after removing "reuters"):

| Pushes toward | Top words |
|---|---|
| Real | `said`, `on`, `washington`, `in`, `wednesday`, `republican`, `thursday`, `tuesday`, `told`, `minister` |
| Fake | `via`, `us`, `this`, `read`, `mr`, `just`, `that`, `gop`, `com`, `obama` |

4. **My own tests.** I wrote three short texts and ran them through the model:

| Text | Truth | Model said |
|---|---|---|
| True fact, news-agency style | real | real |
| Same true fact, clickbait style | real | **fake** (wrong) |
| False claim, news-agency style | fake | **real** (wrong) |

Only three texts, so this is a strong hint rather than proof.

## What I would do next

- Train on real and fake articles from many different sources.
- Strip style markers (datelines like "WASHINGTON -", "via", "read more") during cleaning.
- Evaluate on articles from outside this dataset, not only on the held-out split.

## Tech stack

Python, pandas, NumPy, scikit-learn, kagglehub, Google Colab.

## Author

Malik Zain ([@Malik-Zain-555](https://github.com/Malik-Zain-555))
