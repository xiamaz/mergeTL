"""
Target models for merged FCS datasets with TL
Set weights from the given base model, freeze layers if necessary
tune parameters where necessary - epochs, learning rate, global decay
"""
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from keras import models
import keras

from flowcat import utils, classifier, io_functions
from flowcat.constants import GROUPS
from flowcat.classifier import som_dataset
from flowcat.flowcat_api import train_som_classifier, prepare_classifier_train_dataset
from flowcat.classifier import SOMClassifier, SOMSaliency, SOMClassifierConfig, create_model_multi_input
from flowcat.classifier.predictions import generate_all_metrics

from transfer_learning_model import create_model


def run_denovo(options, train_dataset, validate_dataset):
    config = options["config"]

    model = train_som_classifier(train_dataset, validate_dataset, config)

    output = utils.URLPath(options["output_path"])

    if validate_dataset:
        validate = model.create_sequence(validate_dataset, config.valid_batch_size)

        pred_arr, pred_labels = model.predict_generator(validate)
        true_labels = validate.true_labels
        pred_df = pd.DataFrame(pred_arr, columns=validate.binarizer.classes_, index=validate.dataset.labels)
        io_functions.save_csv(pred_df, output / "preds.csv")
        io_functions.save_json({"true": list(true_labels), "pred": list(pred_labels)}, output / "preds_labels.json")
        generate_all_metrics(true_labels, pred_labels, config.mapping, output)

    model.save(output)
    model.save_information(output)

    keras.backend.clear_session()
    del model


def create_kfold_split(dataset, k_number, stratified=False) -> [("SOMDataset", "SOMDataset")]:
    """Create a list of train test pairs."""
    if stratified:
        group_lists = defaultdict(list)
        for somsample in dataset.data:
            group_lists[somsample.group].append(somsample)

        if any(len(v) < k_number for v in group_lists.values()):
            group_counts = {k: len(v) for k, v in group_lists.items()}
            raise ValueError("K number {k_number} smaller than: {group_counts}")

        group_splits = []
        for group, samples in group_lists.items():
            group_splits.append(np.array_split(pd.Series(samples).sample(frac=1), k_number))

        splits = []
        for n in range(k_number):
            splits.append(pd.concat([v[n] for v in group_splits]))
    else:
        splits = np.array_split(dataset.data.sample(frac=1), k_number)

    split_datasets = []
    for n in range(k_number):
        validate_dataset = som_dataset.SOMDataset(splits[n].reset_index(drop=True), dataset.config)
        train_dataset = som_dataset.SOMDataset(
            pd.concat(splits[:n] + splits[n+1:]).reset_index(drop=True), dataset.config)
        split_datasets.append((train_dataset, validate_dataset))

    return split_datasets


def run_kfold(*, output_path, som_dataset_path, k_number=5, panel="MLL", rerun=False, stratified=False,):
    if not rerun and output_path.exists():
        LOGGER.info("Existing results exist at %s skipping", output_path)
        return

    args = locals()
    io_functions.save_json(args, output_path / "params.json")

    # set the groups according to the panel
    if panel == "MLL":
        groups = GROUPS
    else:
        groups = ["CLL", "MCL", "LPL", "MZL", "FL", "HCL", "normal"]

    # tubes to be processed for merged samples
    tubes = ("1")

    mapping = {"groups": groups, "map": None}

    dataset = som_dataset.SOMDataset.from_path(som_dataset_path)
    LOGGER.info("Full dataset %s", dataset.group_count)

    splits = create_kfold_split(dataset, k_number=k_number, stratified=stratified)

    for n, (train_dataset, validate_dataset) in enumerate(splits):
        LOGGER.info(f"SPLIT n={n}")
        LOGGER.info("Train dataset %s", train_dataset.group_count)
        LOGGER.info("Validation dataset %s", validate_dataset.group_count)

        # change epochs to suit each dataset
        options = {
            "output_path": output_path / f"kfold_n{n}",
            "config": classifier.SOMClassifierConfig(**{
                "tubes": {tube: dataset.config[tube] for tube in tubes},
                "groups": groups,
                "pad_width": 2,
                "mapping": mapping,
                "cost_matrix": None,
                "train_epochs": 20,
            })
        }
        run_denovo(options, train_dataset, validate_dataset)


if __name__ == "__main__":
    OUTPUT = utils.URLPath("/data/flowcat-data/2020-12_kfold_n10_denovo")
    LOGGER = utils.logs.setup_logging(OUTPUT / "logs.txt", "merged model with TL")
    experiments = {
        "mll5f": {
            "output_path": OUTPUT / "mll5f",
            "som_dataset_path": "/data/flowcat-data/2020_Nov_rerun/Merged_SOM/MLL5F",
            "panel": "MLL",
            "k_number": 10,
            "rerun": False,
            "stratified": False,
        },
        "bonn": {
            "output_path": OUTPUT / "bonn",
            "som_dataset_path": "/data/flowcat-data/2020_Nov_rerun/Merged_SOM/Bonn/with_9F_ref",
            "panel": "BONN",
            "k_number": 10,
            "rerun": False,
            "stratified": False,
        },
        "berlin": {
            "output_path": OUTPUT / "berlin",
            "som_dataset_path": "/data/flowcat-data/2020_Nov_rerun/Merged_SOM/Berlin",
            "panel": "BERLIN",
            "k_number": 10,
            "rerun": False,
            "stratified": False,
        },
    }

    for name, config in experiments.items():
        config = {k: utils.URLPath(v) if k.endswith("path") else v for k, v in config.items()}
        LOGGER.info(f"Running {name}")
        run_kfold(**config)
        LOGGER.info(f"Completed {name}")
