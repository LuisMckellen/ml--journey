## The same code gave two different scores

While recording the project's first leakage-free metric, the baseline
notebook and the training script disagreed:

| Environment | xgboost | macro F1 |
|---|---|---|
| Kaggle notebook | 3.2.0 | 0.6015 |
| Local venv (`src/train.py`) | 3.4.1 | **0.5613** |

Four points of macro F1 between two runs that should have been
identical.

### What was ruled out

Everything about the data and the code was the same in both:

- same CSV, 3276 rows, no duplicates dropped
- same split — `test_size=0.2, random_state=42, stratify=y`, giving
  2620 train / 656 test in both environments
- same `random_state=42` on the model
- `eval_metric="logloss"` — reports during training, doesn't change the
  fitted trees, so it can't move the score

### What it was

The library version. Kaggle's image runs xgboost 3.2.0; the local venv
runs 3.4.1. Default hyperparameters changed between those releases, so
"XGBClassifier with default settings" describes two different models
depending on where it runs.

### What changed as a result

- The defaults were pinned down rather than assumed. Setting
  `n_estimators=100, max_depth=6, learning_rate=0.3` explicitly
  reproduces 0.5613 exactly, which confirms those are 3.4.1's defaults
  and that the score comes from the values, not from anything implicit.
  `src/train.py` still relies on the defaults; the values are recorded
  here so the model is reconstructable if a future version moves them.
- The reported metric is recorded together with the version that
  produced it. **macro F1 0.5613, xgboost 3.4.1** is the claim; the
  number alone isn't one.
- `models/model.json` is written with `save_model` rather than pickled.
  A pickled booster is tied to the version that created it; the JSON
  format loads across versions.
- `/ping` returns the running xgboost version, so the deployed
  container can be checked against the environment the metric came
  from.

The point of containerising this project stopped being abstract at
that moment. "It works on my machine" is usually a slogan; here it was
a measured 4-point difference in the headline number, found the day
before the Docker work started.
