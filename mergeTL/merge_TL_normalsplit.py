"""
Target models for merged FCS datasets with TL
Set weights from the given base model, freeze layers if necessary
tune parameters where necessary - epochs, learning rate, global decay
"""
import os
import sys
import random
import pickle

import pandas as pd
from keras import layers, regularizers, models

from flowcat import utils, classifier, io_functions
from flowcat.constants import GROUPS
from flowcat.classifier import som_dataset
from flowcat.flowcat_api import train_som_classifier, prepare_classifier_train_dataset
from flowcat.classifier import SOMClassifier, SOMSaliency, SOMClassifierConfig, create_model_multi_input
from flowcat.classifier.som_dataset import SOMDataset


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
        units=yshape, activation="softmax"
    )(x)

    model = models.Model(inputs=inputs, outputs=x)
    return model


def print_usage():
    """print syntax of script invocation"""
    print("\nUsage:")
    print("python {0:} SOM_datapath outputpath panel(Bonn, MLL,"
          "or Berlin) basemodel_path\n".format(
        os.path.basename(sys.argv[0])))
    return


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print_usage()
        raise Exception("Invalid arguments")

    SOM_DATASET = utils.URLPath(sys.argv[1])
    OUTPUT = utils.URLPath(sys.argv[2])
    PANEL = sys.argv[3]
    BASE_MODEL_PATH = utils.URLPath(sys.argv[4])
    EPOCHS = int(sys.argv[5])

    LOGGER = utils.logs.setup_logging(None, "merged model with TL")

    # set the groups according to the panel
    if PANEL == "MLL":
        groups = GROUPS
    else:
        groups = ["CLL", "MCL", "LPL", "MZL", "FL", "HCL", "normal"]

    # tubes to be processed for merged samples
    tubes = ("1")

    mapping = None

    dataset = som_dataset.SOMDataset.from_path(SOM_DATASET)

    train_dataset, validate_dataset = prepare_classifier_train_dataset(dataset, groups=groups, mapping=mapping,
                                                                       balance=None)

    print(train_dataset.group_count)
    print(validate_dataset.group_count)

    # resplit normal in train and validate
    normal_ratio = 0.25
    non_normals = [d for d in train_dataset.data if d.group != "normal"]
    normals = [d for d in train_dataset.data if d.group == "normal"]
    random.shuffle(normals)
    train_data = non_normals + normals[:int(len(normals) * normal_ratio)]
    validate_dataset.data = pd.Series(list(validate_dataset.data)+normals[int(len(normals) * normal_ratio):])
    train_dataset = SOMDataset(pd.Series(train_data), train_dataset.config)

    print(train_dataset.group_count)
    print(validate_dataset.group_count)

    # change epochs to suit each dataset
    config = classifier.SOMClassifierConfig(**{
        "tubes": {tube: dataset.config[tube] for tube in tubes},
        "groups": groups,
        "pad_width": 2,
        "mapping": mapping,
        "cost_matrix": None,
        "train_epochs": EPOCHS,
    })

    # load base model and get weights
    base_model = models.load_model(str(BASE_MODEL_PATH / "model.h5"))
    weights = base_model.get_weights()

    # create model
    model = create_model(config.inputs, config.output)

    model.set_weights(weights)

    # freeze 2 dense layers: check for each dataset
    model.get_layer('dense_1').trainable = False
    model.get_layer('dense_2').trainable = False

    model.compile(
        loss=config.get_loss(modeldir=None),
        optimizer="adam", metrics=["accuracy"])

    # cast to SOMConfig instance
    model = SOMClassifier(config, model)

    train = model.create_sequence(train_dataset, config.train_batch_size)

    if validate_dataset is not None:
        validate = model.create_sequence(validate_dataset, config.valid_batch_size)
    else:
        validate = None

    model.train_generator(train, validate, epochs=config.train_epochs, class_weight=None)

    model.save(OUTPUT)
    model.save_information(OUTPUT)

    for i, (_, history) in enumerate(model.training_history):
        hist_output = OUTPUT / f"history_{i}.pb"
        with open(str(hist_output), "wb") as f:
            pickle.dump(history.history, f)
