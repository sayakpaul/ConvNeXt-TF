import ml_collections


def convnext_tiny_config() -> ml_collections.ConfigDict:
    configs = ml_collections.ConfigDict()
    configs.depths = [3, 3, 9, 3]
    configs.dims = [96, 192, 384, 768]
    return configs


def convnext_small_config() -> ml_collections.ConfigDict:
    configs = convnext_tiny_config()
    configs.depths = [3, 3, 27, 3]
    return configs


def convnext_base_config() -> ml_collections.ConfigDict:
    configs = convnext_small_config()
    configs.dims = [128, 256, 512, 1024]
    return configs


def convnext_large_config() -> ml_collections.ConfigDict:
    configs = convnext_base_config()
    configs.dims = [192, 384, 768, 1536]
    return configs


def convnext_xlarge_config() -> ml_collections.ConfigDict:
    configs = convnext_large_config()
    configs.dims = [256, 512, 1024, 2048]
    return configs


def get_model_config(model_name: str) -> ml_collections.ConfigDict:
    if model_name == "convnext_tiny":
        return convnext_tiny_config()
    elif model_name == "convnext_small":
        return convnext_small_config()
    elif model_name == "convnext_base":
        return convnext_base_config()
    elif model_name == "convnext_large":
        return convnext_large_config()
    else:
        return convnext_xlarge_config()
