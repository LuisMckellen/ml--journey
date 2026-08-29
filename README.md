# Water Potability — Prediction API

**Live:** https://ml-journey-mpqh.onrender.com · [Docs](https://ml-journey-mpqh.onrender.com/docs)

A binary classifier for drinking-water safety, served as a FastAPI
endpoint and deployed as a container. Nine water-quality measurements
in, a potability prediction and calibrated probability out.

> Free-tier hosting sleeps after 15 minutes of inactivity. The first
> request may take ~50 seconds to wake the instance.

---

## The dataset

3,276 samples, nine numeric features, binary `Potability` target.
**61% not potable / 39% potable** — imbalanced enough that accuracy is
misleading.

Three columns have missing values: `ph`, `Sulfate`, `Trihalomethanes`.
These are imputed with medians **learned from the training split only**.

## The leakage bug

The first working version computed medians on the full dataframe and
then split. That leaks the test set's distribution into training: the
imputed training values carry information about data the model is
supposed to have never seen. It inflates the score without improving the
model.

Found by reading the code, not by a failing test — the pipeline ran
cleanly and produced a plausible number the whole time.

The fix is an ordering change:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
medians = X_train.median()      # train only
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)
```

The old reported metric (macro F1 0.598) came from the leaky pipeline
and has been retired. Everything below is post-fix.

## Choosing the metric

**Accuracy is wrong here.** 61% of samples are "not potable," so a model
that predicts that label for every sample scores 61% and has learned
nothing.

**Recall alone is gameable.** A false negative — calling unsafe water
safe — is the costly error, which makes recall tempting. But recall of
*which* class? A dummy classifier predicting "not potable" every time
gets recall₀ = 1.0000 for free while never identifying a single safe
sample.

That isn't hypothetical. Logistic regression did exactly this:

| Model | macro F1 | recall₀ | recall₁ |
|---|---|---|---|
| LogisticRegression | 0.3788 | 1.0000 | 0.0000 |
| RandomForest | 0.5887 | 0.8925 | 0.3047 |
| XGBoost | 0.6015 | 0.8050 | 0.3984 |

*Baseline comparison run in a notebook environment; see the version note
below for why the XGBoost figure differs from the production metric.*

Accuracy would have called that logistic regression 61% correct. Macro
F1 — F1 per class, averaged unweighted — scores it 0.3788, because a
model cannot score well on macro F1 without doing something useful on
both classes.

**Macro F1 is the metric used from here on.**

## Results

Production model, `src/train.py`, xgboost 3.4.1:

| | precision | recall | f1 | support |
|---|---|---|---|---|
| 0 — not potable | 0.6520 | 0.8150 | 0.7244 | 400 |
| 1 — potable | 0.5256 | 0.3203 | 0.3981 | 256 |
| **macro avg** | 0.5888 | 0.5677 | **0.5613** | 656 |

Accuracy 0.6220, against a 61% majority-class baseline.

**The number with real-world cost is recall₀ = 0.8150.** The model
catches 82% of unsafe samples; the 18% it misses are unsafe water called
safe, which is the error that matters. Everything else is bookkeeping by
comparison.

That is also why the metric isn't recall₀. Optimising for it directly
has a trivial degenerate solution — predict "not potable" always, score
1.0000, help nobody. Macro F1 makes that impossible: a model cannot
score well on it without discriminating between the classes, so
protecting class 0 has to be earned rather than declared.

Which is what recall₁ = 0.3203 is really measuring. Not that
potable-detection is the goal, but that the model is leaning heavily on
the majority class, and a classifier with recall₀ = 0.99 and recall₁ =
0.05 would look safe while being useless.

`scale_pos_weight` is the parameter that shifts this balance, and W3
applies it as a question rather than a fix: does raising recall₁ improve
macro F1, and what does it cost recall₀? If macro F1 rises while recall₀
falls to 0.70, that trade should be rejected — the aggregate metric
improving is not worth missing more unsafe water. The before-values are
recorded above so the comparison can be made on numbers.

A single 656-row test split is noisy, so none of these figures have
error bars yet. Cross-validation is next; until then, no hyperparameter
tuning — trying configurations against one test set and keeping the
winner fits the test set, which is the same category of mistake as the
leakage bug.

## The same code gave two different scores

The baseline notebook reported macro F1 **0.6015**. `src/train.py`
reported **0.5613**. Same CSV, same 2620/656 stratified split, same
`random_state=42`, no duplicates dropped.

Ruled out in order: the split parameters, the model seed, and
`eval_metric` (which reports during training but doesn't change the
fitted trees).

The cause was the environment. The notebook ran on **xgboost 3.2.0**;
the local venv runs **3.4.1**. Default hyperparameters changed between
those releases, so "XGBClassifier with defaults" describes two different
models depending on where it runs.

What changed as a result:

- **The metric is reported with the version that produced it.** *macro F1
  0.5613, xgboost 3.4.1* is the claim. The number alone isn't one.
- **The defaults were pinned down rather than assumed.** Setting
  `n_estimators=100, max_depth=6, learning_rate=0.3` explicitly
  reproduces 0.5613 exactly, confirming those are 3.4.1's defaults.
- **The model is saved with `save_model`, not pickled.** A pickled
  booster is tied to the xgboost version that wrote it; `model.json`
  loads across versions.
- **`/ping` reports the running xgboost version**, so a deployed
  instance can be checked against the environment the metric came from.

Verified end to end — the same request returns `0.1700534224510193` from
local uvicorn, from the Docker container, and from the live Render
deployment.

## API

`POST /predict`

```json
{
  "ph": 7.0, "Hardness": 200, "Solids": 20000,
  "Chloramines": 7, "Sulfate": 300, "Conductivity": 400,
  "Organic_carbon": 15, "Trihalomethanes": 60, "Turbidity": 4
}
```

```json
{
  "potability": 0,
  "probability_potable": 0.1700534224510193,
  "imputed_fields": []
}
```

All nine fields are optional. Omitted fields are filled with the
training medians and named in `imputed_fields`, so a caller can always
see which parts of a prediction came from their data and which came from
the population. Unknown fields are rejected (`extra="forbid"`).

Note the response above: nine entirely plausible mid-range readings, and
the model returns 17% potable. The classifier's lean toward "not
potable" is visible in a single call — conservative on safety, and
correspondingly unwilling to clear anything.

`GET /ping` — liveness, artifact status, xgboost version
`GET /docs` — interactive OpenAPI

Validation returns 422 with a specific message: out-of-range pH,
negative concentrations, an all-null body, or unexpected fields.
If the model artifacts are absent the API still starts and returns 503
naming the missing files, rather than crashing on import.

## Running it

```bash
git clone https://github.com/LuisMckellen/ml--journey
cd ml--journey
pip install -r requirements.txt
python -m src.train          # writes models/
uvicorn app.main:api --reload
```

Docker:

```bash
docker build -t water-api .
docker run -p 8000:8000 water-api
```

Image: 472MB. `.dockerignore` keeps the virtualenv, notebooks, and raw
data out of the build context — `data/` isn't copied because the image
serves a trained model rather than training one.

Tests:

```bash
pytest -v
```

Five tests covering the happy path, missing-field imputation, and four
validation cases.

## Stack

Python 3.14 · XGBoost 3.4.1 · FastAPI · pandas · pytest · Docker · Render

## Next

- 5-fold cross-validation — a mean and a spread instead of one number
- `scale_pos_weight` — does raising recall₁ improve macro F1, and what
  does it cost recall₀?
- SHAP feature importance
- GitHub Actions running pytest on every push
