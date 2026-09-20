# Machine Learning Classification: Bank Marketing and Persian News

Coursework project covering two supervised-classification problems:

1. Predicting whether a bank client will subscribe to a term deposit.
2. Classifying Persian news headlines as `Economy`, `Sport`, or `Tech` with a
   multinomial Naive Bayes classifier implemented from scratch.

The repository includes cleaned notebooks and the Bank Marketing dataset. The
Persian News dataset is not redistributed because its original source and
redistribution license were not provided with the course materials. The notebooks
use fixed random seeds and repository-relative paths so they can be run on another
machine.

## Project highlights

- Exploratory analysis for mixed numeric and categorical bank-marketing data.
- Training-only correlation filtering and Isolation Forest outlier detection.
- Ordinal and frequency encoding, standardization, and SMOTE for class imbalance.
- Decision Tree, linear SVM, Logistic Regression, KNN, and XGBoost comparison.
- SMOTE is applied within each cross-validation training fold during hyperparameter
  search.
- Persian normalization, stopword removal, vocabulary analysis, and a custom
  log-space multinomial Naive Bayes implementation with Laplace smoothing.
- Confusion matrices, classification reports, ROC curves, per-class F1 scores,
  frequent-word plots, and optional Persian word clouds.

## Repository structure

```text
.
├── data/
│   ├── bank_marketing.csv
│   └── README.md
├── notebooks/
│   ├── bank_marketing_classification.ipynb
│   └── persian_news_naive_bayes.ipynb
├── tests/
│   └── test_project_integrity.py
├── .github/workflows/quality.yml
├── CITATION.cff
├── LICENSE
├── NOTICE.md
└── requirements.txt
```

## Datasets

### Bank Marketing

`data/bank_marketing.csv` contains 41,188 observations, 20 input features, and
the binary target `y`. The positive class is imbalanced: 4,640 rows are `yes`
and 36,548 are `no`.

This is the `bank-additional-full.csv` variant of the UCI Bank Marketing
dataset. It is distributed under CC BY 4.0. Cite:

> Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI
> Machine Learning Repository. https://doi.org/10.24432/C5K306

Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing

The `duration` field is only known after a call ends. Keep it for reproducing
the coursework, but remove it when evaluating a model intended to score clients
before a call.

### Persian News

The coursework dataset contains 61,843 headlines with two columns: `Text` and
`Topic`. The class distribution is:

| Topic | Rows |
| --- | ---: |
| Economy | 20,282 |
| Sport | 21,000 |
| Tech | 20,561 |

This dataset was supplied by the course teaching assistant. Its original
publication source and redistribution license were not included with the course
materials, so it is **not redistributed in this public repository**.

To run the Persian-news notebook, place an authorized local copy at:

```text
data/persian_news.csv
```

See `data/README.md` for the expected schema.

## Setup

Python 3.11 is recommended.

```bash
git clone https://github.com/maryamjbr/ml-classification-bank-marketing-persian-news.git
cd ml-classification-bank-marketing-persian-news

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebooks from the `notebooks/` directory and run their cells from top
to bottom.

The Persian word-cloud cells look for `assets/Vazirmatn-Regular.ttf` first. If
that font is absent, the classifier and metric plots still run, while word-cloud
generation is skipped with an explanatory message.

## Notebook workflow

### Bank marketing classification

1. Load and profile the semicolon-delimited data.
2. Explore distributions, outliers, categorical relationships, and correlations.
3. Create a stratified 70/30 train/test split.
4. Learn correlation filtering and Isolation Forest filtering from training data
   only; the test set remains untouched.
5. Encode categorical variables and standardize features using training data.
6. Compare baseline classifiers after training-set SMOTE resampling.
7. Tune each model with stratified five-fold cross-validation and fold-local
   SMOTE, then evaluate the selected estimators on the held-out test set.

The full SVM and XGBoost grids are computationally expensive. Runtime depends on
CPU and available memory.

### Persian news Naive Bayes

1. Split the raw headlines with stratification.
2. Normalize Arabic/Persian character variants and remove URLs, email addresses,
   digits, Latin letters, punctuation, repeated characters, and stopwords.
3. Count per-class word frequencies and calculate class priors.
4. Predict in log space with Laplace smoothing. Tokens outside the training
   vocabulary are ignored.
5. Evaluate on the held-out test set and visualize per-class behavior.

## Reproducibility and validation

- Randomized operations use `random_state=42`.
- Notebook outputs are intentionally cleared to keep the repository small and
  prevent local paths and environment warnings from being committed.
- The included integrity tests validate the Bank Marketing dataset, notebook JSON,
  Python cell syntax, and portable paths.
- If a local authorized copy of `data/persian_news.csv` is present, its schema,
  row count, and labels are also validated.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

## Author

- Maryam Jabbari

## License

Original source code and repository documentation are available under the MIT
License. Third-party datasets have separate terms; see [NOTICE.md](NOTICE.md).
