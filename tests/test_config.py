from .context import Doc, config_context


def test_default():
    d = Doc("Ali topu tut. Ömer ılık süt iç.")

    assert Doc.config['tf']['method'] == "log_norm"
    assert d.builder.config['tf']['method'] == "log_norm"


def test_config_context():
    for tf in ["binary", "raw"]:
        with config_context(tf__method=tf) as Doc_with_context:
            d = Doc_with_context("Ali topu tut. Ömer ılık süt iç.")

            assert Doc_with_context.config['tf']['method'] == tf
            assert d.builder.config['tf']['method'] == tf
            assert all(s.tf_method == tf for s in d)
