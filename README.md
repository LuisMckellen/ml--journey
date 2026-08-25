# ml--journey

ML work as a first-year CSE student. One project taken properly from raw data to a deployed service, instead of several that stop at the notebook.

## Water Potability

Predicting whether water is safe to drink from nine chemical measurements. 3,276 samples, Kaggle.

**Where it's at:** EDA done, baseline models compared, FastAPI skeleton running. Next up is SHAP, then Docker and getting it deployed.

Current best: XGBoost, macro F1 0.598.

### What I found

No feature correlates with the target above 0.05. I expected that to mean linear models would fail, so I checked — LogisticRegression scored exactly the same as a dummy classifier that always predicts "not potable". Accuracy 0.610, recall 0.000. So that held up.

Picking a metric took longer than picking a model. Accuracy is useless here since 61% of the data is one class. Recall on the unsafe class seemed right — until the dummy scored a perfect 1.000 on it by labelling everything unsafe. F1 on that class put the dummy at 0.758 and RandomForest at 0.760, which is noise.

Macro F1 was the first one where anything separated:

| | Dummy | Best model |
|---|---|---|
| Accuracy | 0.610 | 0.659 |
| Recall (not potable) | 1.000 | 0.887 |
| F1 (not potable) | 0.758 | 0.760 |
| Macro F1 | 0.379 | 0.598 |

The habit I took from this: check what a dummy scores before trusting any metric.

I also shuffled the training labels and retrained, to see whether the models were learning anything real. Macro F1 dropped about 0.119 for both — consistent across RandomForest and XGBoost. So there's signal, just not much of it.

The metric matters because the errors aren't symmetric. Unsafe water labelled safe means someone drinks it. Safe water labelled unsafe just wastes it.

### Notebooks

- `water-potability-eda.ipynb` — missing values, imputation, correlation, class balance
- `water-potability-baseline.ipynb` — four models, metric selection, shuffle test
- `water-potability-api.ipynb` — FastAPI, tested with TestClient

### Still open

- Only one train/test split so far. XGBoost beats RandomForest by 0.014 macro F1, which a different random seed could flip. Needs cross-validation.
- Haven't tried `scale_pos_weight` or `class_weight='balanced'` yet.
- LogisticRegression threw a convergence warning — `Solids` goes to 61,000 while `Turbidity` stays under 7. Scaling would silence it, but I don't know yet whether it changes the result.
- `ph` runs exactly 0.00 to 14.00. Real water doesn't hit either end, so the data may be synthetic or clipped.

## Stack

Python, pandas, scikit-learn, XGBoost, FastAPI
