from tqdm import tqdm
import os

"""
Details about these checkpoints are available here:
https://github.com/facebookresearch/ConvNeXt#results-and-pre-trained-models.
"""

imagenet_1k_224 = {
    "convnext_tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
}

imagenet_1k_384 = {
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth",
}

imagenet_21k_224 = {
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

imagenet_21k_1k_224 = {
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_224.pth",
    "convnext_xlarge": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_224_ema.pth",
}

imagenet_21k_1k_384 = {
    "convnext_base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_384.pth",
    "convnext_large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_1k_384.pth",
    "convnext_xlarge": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_1k_384_ema.pth",
}

print("Converting 224x224 resolution ImageNet-1k models.")
for model in tqdm(imagenet_1k_224):
    print(f"Converting {model}.")
    command = f"python convert.py -m {model} -c {imagenet_1k_224[model]}"
    os.system(command)


print("Converting 384x384 resolution ImageNet-1k models.")
for model in tqdm(imagenet_1k_384):
    print(f"Converting {model}.")
    command = f"python convert.py -m {model} -c {imagenet_1k_384[model]} -r 384"
    os.system(command)


print("Converting 224x224 resolution ImageNet-21k models.")
for model in tqdm(imagenet_21k_224):
    print(f"Converting {model}.")
    command = f"python convert.py -d imagenet-21k -m {model} -c {imagenet_21k_224[model]} -r 224"
    os.system(command)


print(
    "Converting 224x224 resolution ImageNet-21k trained ImageNet-1k fine-tuned models."
)
for model in tqdm(imagenet_21k_1k_224):
    print(f"Converting {model}.")
    command = f"python convert.py -m {model} -c {imagenet_21k_1k_224[model]} -r 224"
    os.system(command)


print(
    "Converting 384x384 resolution ImageNet-21k trained ImageNet-1k fine-tuned models."
)
for model in tqdm(imagenet_21k_1k_384):
    print(f"Converting {model}.")
    command = f"python convert.py -m {model} -c {imagenet_21k_1k_384[model]} -r 384"
    os.system(command)
