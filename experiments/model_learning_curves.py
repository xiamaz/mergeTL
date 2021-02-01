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


def main(args):
    dataset = som_dataset.SOMDataset.from_path(args.input)
    val = args.val
    train = args.train
    OUTPUT = args.output
    PANEL = args.panel
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
                                               "train_epochs": 20,
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

    model = fc_api.train_som_classifier(train_dataset, validate_dataset, config)

    model.save(OUTPUT)
    model.save_information(OUTPUT)


if __name__ == "__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--input", type=utils.URLPath)
    PARSER.add_argument("--output", type=utils.URLPath)
    PARSER.add_argument("--val", type=utils.URLPath)
    PARSER.add_argument("--train", type=utils.URLPath)
    PARSER.add_argument("--panel", type=str)
    PARSER.add_argument("--bal", type=int)
    main(PARSER.parse_args())
