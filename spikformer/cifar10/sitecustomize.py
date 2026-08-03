"""Compatibility shims loaded before CIFAR-10 training scripts start."""

try:
    import timm.models as timm_models

    if not hasattr(timm_models, "convert_splitbn_model"):
        def convert_splitbn_model(model, *args, **kwargs):
            return model

        timm_models.convert_splitbn_model = convert_splitbn_model

    if hasattr(timm_models, "create_model"):
        _create_model = timm_models.create_model

        def create_model(*args, **kwargs):
            kwargs.pop("pretrained_cfg", None)
            kwargs.pop("pretrained_cfg_overlay", None)
            return _create_model(*args, **kwargs)

        timm_models.create_model = create_model
except Exception:
    pass
