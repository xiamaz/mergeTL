"""
Test bonn data using previously generated model.
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


def load_model(path):
    binarizer = io_functions.load_joblib(path / "binarizer.joblib")
    model = models.load_model(str(path / "model.h5"))
    return binarizer, model


def main(args):
    dataset = som_dataset.SOMDataset.from_path(args.input)
    val = args.val
    train = args.train
    OUTPUT = args.output

    groups = ["MCL", "PL"]
    tubes = ("1")
    mapping = None
    balance = {
        "MCL": 20,
        "PL": 20,
    }

    config = classifier.SOMClassifierConfig(**{"tubes": {tube: dataset.config[tube] for tube in tubes},
                                               "groups": groups,
                                               "pad_width": 2,
                                               "mapping": mapping,
                                               "cost_matrix": None,
                                               })
    val = io_functions.load_json(val)
    validate_dataset = dataset.filter(labels=val)

    labels = io_functions.load_json(train)
    train_dataset = dataset.filter(labels=labels)



    train_dataset, validate_dataset = fc_api.prepare_classifier_train_dataset(train_dataset,split_ratio=0.9,
                                                                              groups=groups,
                                                                              mapping=mapping,
                                                                              balance=balance, val_dataset = validate_dataset)

    print(train_dataset.group_count)
    print(validate_dataset.group_count)

    binarizer, model = load_model(args.model)

    trainseq = som_dataset.SOMSequence(train_dataset, binarizer, tube=config.tubes, pad_width=config.pad_width)
    validseq = som_dataset.SOMSequence(validate_dataset, binarizer, tube=config.tubes, pad_width=config.pad_width)

    


    model.fit_generator(
        generator=trainseq,
        epochs=10,
        validation_data=validseq)

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
    PARSER.add_argument("--val", type=utils.URLPath)
    PARSER.add_argument("--train", type=utils.URLPath)
    PARSER.add_argument("model", type=utils.URLPath)
    main(PARSER.parse_args())
