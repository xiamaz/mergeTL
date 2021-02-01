"""
Generate learning curve for various training size for Target models without any TL
same hyper parameters as the base model
Use same validation set for every iteration and subsample from a given training set

"""
from argparse import ArgumentParser

from flowcat import classifier, utils, io_functions
from flowcat.constants import DEFAULT_CLASSIFIER_CONFIG, GROUPS, DEFAULT_CLASSIFIER_ARGS
from flowcat import flowcat_api as fc_api
from flowcat.classifier import som_dataset
from flowcat.classifier import SOMClassifier, SOMSaliency, SOMClassifierConfig, create_model_multi_input
from keras import layers, regularizers, models


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


def main(args):
    dataset = som_dataset.SOMDataset.from_path(args.input)
    val = args.val
    train = args.train
    OUTPUT = args.output
    PANEL = args.panel
    basemodel = args.basemodel
    bal = args.bal

    # set the groups according to the panel
    if PANEL == "MLL":
        groups = GROUPS
    elif PANEL == "ERLANGEN":
         groups = ["CLL", "MBL", "MCL", "LPL", "MZL", "FL", "HCL", "normal"]
    else:
        groups = ["CLL", "MCL", "LPL", "MZL", "FL", "HCL", "normal"]

    tubes = ("1")
    mapping = None

    balance = dict((key, bal) for key in groups)

    config = classifier.SOMClassifierConfig(**{"tubes": {tube: dataset.config[tube] for tube in tubes},
                                               "groups": groups,
                                               "pad_width": 2,
                                               "mapping": mapping,
                                               "cost_matrix": None,
                                               "train_epochs": 15,
                                               })
    val = io_functions.load_json(val)
    validate_dataset = dataset.filter(labels=val)

    labels = io_functions.load_json(train)
    train_dataset = dataset.filter(labels=labels)

    train_dataset, validate_dataset = fc_api.prepare_classifier_train_dataset(train_dataset, split_ratio=0.9,
                                                                              groups=groups,
                                                                              mapping=mapping,
                                                                              balance=balance,
                                                                              val_dataset=validate_dataset)

    print(train_dataset.group_count)
    print(validate_dataset.group_count)

    # load base model and get weights
    base_model = models.load_model(str(basemodel / "model.h5"))
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


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--input", type=utils.URLPath)
    PARSER.add_argument("--output", type=utils.URLPath)
    PARSER.add_argument("--val", type=utils.URLPath)
    PARSER.add_argument("--train", type=utils.URLPath)
    PARSER.add_argument("--basemodel", type=utils.URLPath)
    PARSER.add_argument("--panel", type=str)
    PARSER.add_argument("--bal", type=int)

    main(PARSER.parse_args())
