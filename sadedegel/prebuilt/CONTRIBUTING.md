<a href="http://sadedegel.ai"><img src="https://sadedegel.ai/assets/img/logo-2.png" width="125" height="125" align="right" /></a>

# Contribute to sadedeGel Prebuilt Models

- [ ] Use [`Text2Doc`](../extension/sklearn.py) transformation in pipeline
- [ ] Prefer `icu` (default) word tokenizer to ensure your model will work with core sadedegel.
- [ ] Use an estimator with `predict_proba`
- [ ] Don't use cv in reporting your accuracy ([README](README.md)) if you already have a separate test dataset
- [ ] Don't use `partial_fit` if your dataset is small enough.