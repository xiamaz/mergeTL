"""
Train model on Munich 5f panel: CLl and normal
"""
from argparse import ArgumentParser

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.manifold import TSNE
from keras import layers, regularizers, models

from flowcat import classifier, utils, io_functions
from flowcat.constants import DEFAULT_CLASSIFIER_CONFIG, GROUPS, DEFAULT_CLASSIFIER_ARGS
from flowcat import flowcat_api as fc_api
from flowcat.classifier import som_dataset
from flowcat.classifier.models import create_model_multi_input

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_model(input_shapes, yshape, global_decay=5e-6) -> models.Model:
    """Create a multi input model for keras."""
    segments = []
    inputs = []
    for xshape in input_shapes:
        ix = layers.Input(shape=xshape)
        inputs.append(ix)
        x = layers.Conv2D(
            filters=32, kernel_size=4, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(ix)
        x = layers.Conv2D(
            filters=48, kernel_size=3, activation="relu", strides=1,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        # x = layers.Conv2D(
        #     filters=48, kernel_size=2, activation="relu", strides=1,
        #     kernel_regularizer=regularizers.l2(global_decay),
        # )(x)
        x = layers.Conv2D(
            filters=64, kernel_size=2, activation="relu", strides=2,
            kernel_regularizer=regularizers.l2(global_decay),
        )(x)
        # x = layers.MaxPooling2D(pool_size=2, strides=2)(x)

        # x = layers.GlobalAveragePooling2D()(x)
        x = layers.GlobalMaxPooling2D()(x)
        segments.append(x)

    if len(segments) > 1:
        x = layers.concatenate(segments)
    else:
        x = segments[0]

    x = layers.Dense(
        units=64, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)
    x = layers.Dense(
        units=32, activation="relu",
        # kernel_initializer="uniform",
        kernel_regularizer=regularizers.l2(global_decay)
    )(x)

    x = layers.Dense(
        units=yshape, activation="sigmoid"
    )(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model


def main(args):
    MLL5F = som_dataset.SOMDataset.from_path(args.input)
    OUTPUT = args.output
    #val_labels = args.val
    #train_labels = args.train
    #labels = args.labels
    LOGGER = utils.logs.setup_logging(None, "classify")

    groups = ["MCL", "PL"]
    tubes = ("1")
    mapping = None
    balance = {
        "MCL": 20,
        "PL": 20,
    }

    #vallabels = io_functions.load_json(val_labels)
    #validate_dataset = MLL5F.filter(labels=vallabels)

    #labels = io_functions.load_json(train_labels)
    #train_dataset = MLL5F.filter(labels=labels)

    #labels = io_functions.load_json(labels)
    #train_dataset = MLL5F.filter(labels=labels)

   
    train_dataset, validate_dataset = fc_api.prepare_classifier_train_dataset(
        MLL5F,
        split_ratio=0.90,
        groups=groups,
        mapping=mapping,
        balance=None)#, val_dataset = validate_dataset)

    print(train_dataset.group_count)
    print(validate_dataset.group_count)

    config = classifier.SOMClassifierConfig(**{"tubes": {tube: MLL5F.config[tube] for tube in tubes},
                                               "groups": groups,
                                               "pad_width": 2,
                                               "mapping": mapping,
                                               "cost_matrix": None,
                                               })

    model = create_model(config.inputs, 1, global_decay=5e-3)

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "acc",
        ]
    )

    binarizer = LabelBinarizer()
    binarizer.fit(groups)

    trainseq = som_dataset.SOMSequence(train_dataset, binarizer, tube=config.tubes, pad_width=config.pad_width)
    validseq = som_dataset.SOMSequence(validate_dataset, binarizer, tube=config.tubes, pad_width=config.pad_width)

    model.fit_generator(generator=trainseq, validation_data=validseq,
                                epochs=20, shuffle=True, class_weight=None)

    args.output.mkdir(parents=True, exist_ok=True)
    io_functions.save_joblib(binarizer, OUTPUT / "binarizer.joblib")
    model.save(str(args.output / "model.h5"))

    io_functions.save_json(config.to_json(), OUTPUT / "config.json")
    io_functions.save_json(validseq.dataset.labels, OUTPUT / "ids_validate.json")
    io_functions.save_json(trainseq.dataset.labels, OUTPUT / "ids_train.json")


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--input", type=utils.URLPath)
    PARSER.add_argument("--output", type=utils.URLPath)
    #PARSER.add_argument("--val", type=utils.URLPath)
    #PARSER.add_argument("--train", type=utils.URLPath)
    #PARSER.add_argument("--labels", type=utils.URLPath)
    main(PARSER.parse_args())
