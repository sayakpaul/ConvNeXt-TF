"""Generates .tar.gz archives from SavedModels and serializes them."""


from typing import List
import tensorflow as tf
import os


TF_MODEL_ROOT = "gs://convnext/saved_models"
TAR_ARCHIVES = os.path.join(TF_MODEL_ROOT, "tars/")


def generate_fe(model: tf.keras.Model) -> tf.keras.Model:
    """Generates a feature extractor from a classifier."""
    feature_extractor = tf.keras.Model(model.inputs, model.layers[-2].output)
    return feature_extractor


def prepare_archive(model_name: str) -> None:
    """Prepares a tar archive."""
    archive_name = f"{model_name}.tar.gz"
    print(f"Archiving to {archive_name}.")
    archive_command = f"cd {model_name} && tar -czvf ../{archive_name} *"
    os.system(archive_command)
    os.system(f"rm -rf {model_name}")


def save_to_gcs(model_paths: List[str]) -> None:
    """Prepares tar archives and saves them inside a GCS bucket."""
    for path in model_paths:
        print(f"Preparing classification model: {path}.")
        model_name = path.strip("/")
        abs_model_path = os.path.join(TF_MODEL_ROOT, model_name)

        print(f"Copying from {abs_model_path}.")
        os.system(f"gsutil cp -r {abs_model_path} .")
        prepare_archive(model_name)

        print("Preparing feature extractor.")
        model = tf.keras.models.load_model(abs_model_path)
        fe_model = generate_fe(model)
        fe_model_name = f"{model_name}_fe"
        fe_model.save(fe_model_name)
        prepare_archive(fe_model_name)

    os.system(f"gsutil -m cp -r *.tar.gz {TAR_ARCHIVES}")
    os.system("rm -rf *.tar.gz")


model_paths = tf.io.gfile.listdir(TF_MODEL_ROOT)
print(f"Total models: {len(model_paths)}.")

print("Preparing archives for the classification and feature extractor models.")
save_to_gcs(model_paths)
