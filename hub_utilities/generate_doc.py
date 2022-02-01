"""Generates model documentation for ConvNeXt-TF models.

Credits: Willi Gierke
"""

from string import Template
import attr
import os

template = Template(
    """# Module $HANDLE

ConvNeXt model pre-trained on the $DATASET_DESCRIPTION.

<!-- asset-path: https://storage.googleapis.com/convnext/saved_models/tars/$ARCHIVE_NAME.tar  -->
<!-- task: image-classification -->
<!-- network-architecture: convnext -->
<!-- format: saved_model_2 -->
<!-- fine-tunable: true -->
<!-- license: mit -->
<!-- colab: https://colab.research.google.com/github/sayakpaul/ConvNeXt-TF/blob/main/notebooks/classification.ipynb -->

## Overview

This model is a ConvNeXt [1] model pre-trained on the $DATASET_DESCRIPTION. You can find the complete
collection of ConvNeXt models on TF-Hub on [this page](https://tfhub.dev/sayakpaul/collections/convnext/1).

## Using this model

```py
model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/$HANDLE", trainable=False)
])
predictions = model.predict(images)
```

Inputs to the model must:

1. be four dimensional Tensors of the shape `(batch_size, height, width, num_channels)`. Note
that the model expects images with  `channels_last`  property. `num_channels` must be 3. 
2. be resized to $INPUT_RESOLUTION resolution.
3. be normalized with ImageNet-1k statistics.

Please refer to the Colab Notebook to know better.

## Notes

* The original model weights are provided from [2]. There were ported to Keras models
(`tf.keras.Model`) and then serialized as TensorFlow SavedModels. The porting
steps are available in [3].
* The model can be unrolled into a standard Keras model and you can inspect its topology.
To do so, first download the model from TF-Hub and then load it using `tf.keras.models.load_model`
providing the path to the downloaded model folder.

## References

[1] [A ConvNet for the 2020s by Liu et al.](https://arxiv.org/abs/2201.03545)
[2] [ConvNeXt GitHub](https://github.com/facebookresearch/ConvNeXt)
[3] [ConvNeXt-TF GitHub](https://github.com/sayakpaul/ConvNeXt-TF)

## Acknowledgements

* [Vasudev Gupta](https://github.com/vasudevgupta7) 
* [Gus](https://twitter.com/gusthema)
* [Willi](https://ch.linkedin.com/in/willi-gierke)
* [ML-GDE program](https://developers.google.com/programs/experts/)

"""
)


@attr.s
class Config:
    size = attr.ib(type=str)
    dataset = attr.ib(type=str)
    single_resolution = attr.ib(type=int)

    def two_d_resolution(self):
        return f"{self.single_resolution}x{self.single_resolution}"

    def gcs_folder_name(self):
        return f"convnext_{self.size}_{self.dataset}_{self.single_resolution}"

    def handle(self):
        return f"sayakpaul/{self.gcs_folder_name()}/1"

    def rel_doc_file_path(self):
        """Relative to the tfhub.dev directory."""
        return f"assets/docs/{self.handle()}.md"


for c in [
    Config("tiny", "1k", 224),
    Config("small", "1k", 224),
    Config("base", "1k", 224),
    Config("base", "1k", 384),
    Config("large", "1k", 224),
    Config("large", "1k", 384),
    Config("base", "21k_1k", 224),
    Config("base", "21k_1k", 384),
    Config("large", "21k_1k", 224),
    Config("large", "21k_1k", 384),
    Config("xlarge", "21k_1k", 224),
    Config("xlarge", "21k_1k", 384),
    Config("base", "21k", 224),
    Config("large", "21k", 224),
    Config("xlarge", "21k", 224),
]:
    if c.dataset == "1k":
        dataset_text = "ImageNet-1k dataset"
    elif c.dataset == "21k":
        dataset_text = "ImageNet-21k dataset"
    else:
        dataset_text = (
            "ImageNet-21k"
            " dataset and"
            " was then "
            "fine-tuned "
            "on the "
            "ImageNet-1k "
            "dataset"
        )

    save_path = os.path.join(
        "/Users/sayakpaul/Downloads/", "tfhub.dev", c.rel_doc_file_path()
    )
    model_folder = save_path.split("/")[-2]
    model_abs_path = "/".join(save_path.split("/")[:-1])

    if not os.path.exists(model_abs_path):
        os.makedirs(model_abs_path, exist_ok=True)

    with open(save_path, "w") as f:
        f.write(
            template.substitute(
                HANDLE=c.handle(),
                DATASET_DESCRIPTION=dataset_text,
                INPUT_RESOLUTION=c.two_d_resolution(),
                ARCHIVE_NAME=c.gcs_folder_name(),
            )
        )
