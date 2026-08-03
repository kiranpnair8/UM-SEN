"""Compatibility shims loaded before CIFAR-10 training scripts start."""

try:
    import timm.models as timm_models

    if not hasattr(timm_models, "convert_splitbn_model"):
        def convert_splitbn_model(model, *args, **kwargs):
            return model

        timm_models.convert_splitbn_model = convert_splitbn_model
except Exception:
    pass
