# ConvNeXt-TF

This repository provides TensorFlow / Keras implementations of different ConvNeXt
[1] variants. It also provides the TensorFlow / Keras models that have been
populated with the original ConvNeXt pre-trained weights available from [2]. These
models are not blackbox SavedModels i.e., they can be fully expanded into `tf.keras.Model`
objects and one can call all the utility functions on them (example: `.summary()`).

As of today, all the TensorFlow / Keras variants of the models listed
[here](https://github.com/facebookresearch/ConvNeXt#results-and-pre-trained-models)
are available in this repository except for the
[isotropic ones](https://github.com/facebookresearch/ConvNeXt#imagenet-1k-trained-models-isotropic).
This list includes the ImageNet-1k as well as ImageNet-21k models.

Refer to the ["Using the models"](https://github.com/sayakpaul/ConvNeXt-TF#using-the-models)
section to get started.

## Conversion

TensorFlow / Keras implementations are available in `models/convnext_tf.py`.
Conversion utilities are in `convert.py`.

## Models

All the SavedModels are in `gs://convnext/saved_models` bucket. They will be
published from TF-Hub soon. One can query the bucket like so:

```shell
gsutil ls -lh gs://convnext/saved_models
```

There should be a total of 15 different models. You can load any model and get
started like so:

```python
import tensorflow as tf

model_path = "gs://convnext/saved_models/convnext_large_21k_1k_384"
model = tf.keras.models.load_model(model_path)
print(model.summary(expand_nested=True))
```

The model names are interpreted as follows:

* `convnext_large_21k_1k_384`: This means that the model was first pre-trained
on the ImageNet-21k dataset and was then fine-tuned on the ImageNet-1k dataset. 
Resolution used during pre-training and fine-tuning: 384x384.
* `convnext_large_1k_224`: Means that the model was pre-trained on the ImageNet-1k
dataset with a resolution of 224x224.

## Results

Results are on ImageNet-1k validation set (top-1 accuracy). 

| name | original acc@1 | keras acc@1 |
|:---:|:---:|:---:|
| convnext_tiny_1k_224 | 82.1 | 81.312 |
| convnext_small_1k_224 | 83.1 | 82.392 |
| convnext_base_1k_224 | 83.8 | 83.28 |
| convnext_base_1k_384 | 85.1 | 84.876 |
| convnext_large_1k_224 | 84.3 | 83.844 |
| convnext_large_1k_384 | 85.5 | 85.376 |
|  |  |  |
| convnext_base_21k_1k_224 | 85.8 | 85.364 |
| convnext_base_21k_1k_384 | 86.8 | 86.79 |
| convnext_large_21k_1k_224 | 86.6 | 86.36 |
| convnext_large_21k_1k_384 | 87.5 | 87.504 |
| convnext_xlarge_21k_1k_224 | 87.0 | 86.732 |
| convnext_xlarge_21k_1k_384 | 87.8 | 87.68 |

Differences in the results are primarily because of the differences in the library
implementations especially how image resizing is implemented in PyTorch and
TensorFlow. Results can be verified with the code in `i1k_eval`. Logs
are available at [this URL](https://tensorboard.dev/experiment/odN7OPCqQvGYCRpJP1GhRQ/).

## Using the models

* Off-the-shelf classification: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/ConvNeXt-TF/blob/main/notebooks/classification.ipynb)
* Fine-tuning: [Colab Notebook](https://colab.research.google.com/github/sayakpaul/ConvNeXt-TF/blob/main/notebooks/finetune.ipynb)
 
## Upcoming (contributions welcome)

- [ ] Align layer initializers (useful if someone wanted to train the models
from scratch)
- [ ] Allow the models to accept arbitrary shapes (useful for downstream tasks)
- [ ] Convert the [isotropic models](https://github.com/facebookresearch/ConvNeXt#imagenet-1k-trained-models-isotropic) as well 
- [x] Fine-tuning notebook (thanks to [awsaf49](https://github.com/awsaf49))
- [x] Off-the-shelf-classification notebook
- [ ] Publish models on TF-Hub

## References

* ConvNeXt paper: https://arxiv.org/abs/2201.03545
* Official ConvNeXt code: https://github.com/facebookresearch/ConvNeXt

## Acknowledgements

* [Vasudev Gupta](https://github.com/vasudevgupta7) 
* [Gus](https://twitter.com/gusthema)
* [ML-GDE program](https://developers.google.com/programs/experts/)
