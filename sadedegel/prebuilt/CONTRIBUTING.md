<a href="http://sadedegel.ai"><img src="https://sadedegel.ai/assets/img/logo-2.png" width="125" height="125" align="right" /></a>

# Contribute to sadedeGel Prebuilt Models

## Checklist
Please follow the checklist in developing prebuilt NLP models:

- [ ] Use [`Text2Doc`](../extension/sklearn.py) transformation in pipeline
- [ ] Prefer `icu` (default) word tokenizer to ensure your model will work with core sadedegel.
- [ ] Use an estimator with `predict_proba`
- [ ] Don't use cv in reporting your accuracy ([README](README.md)) if you already have a separate test dataset
- [ ] Don't use `partial_fit` if your dataset is small enough.
- [ ] Cite any existing work on the related dataset.
- [ ] Ensure that you use a metric that can be compared with an existing work, if any.
- [ ] Remember to set `pipeline.steps[0][1].Doc` to `None`. Otherwise `Document` instance will be serialized growing pipeline size.
- [ ] Sadedegel provides lots of features. Such as word embeddings. So please don't hesitate to improvise in building models :)