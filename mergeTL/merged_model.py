"""
Train model for merged SOM for base (MLL9F) and target models (MLL5F, Bonn, Berlin)
same parameters as th unmerged model
"""
import os
import sys

from flowcat import utils, classifier, io_functions
from flowcat.constants import GROUPS
from flowcat.classifier import som_dataset
from flowcat.flowcat_api import train_som_classifier, prepare_classifier_train_dataset


def print_usage():
    """print syntax of script invocation"""
    print("\nUsage:")
    print("python {0:} SOM_datapath outputpath panel(Erlangen, Bonn, MLL,"
          "or Berlin)\n".format(
        os.path.basename(sys.argv[0])))
    return


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print_usage()
        raise Exception("Invalid arguments")

    SOM_DATASET = utils.URLPath(sys.argv[1])
    OUTPUT = utils.URLPath(sys.argv[2])
    PANEL = sys.argv[3]

    LOGGER = utils.logs.setup_logging(None, "merged model")

    # set the groups according to the panel
    if panel == "MLL":
        groups = GROUPS
        
    elif panel == "ERLANGEN":
         groups = ["CLL", "MBL", "MCL", "LPL", "MZL", "FL", "HCL", "normal"]
         
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

    # same parameters as the unmerged model for both base and target models
    config = classifier.SOMClassifierConfig(**{
        "tubes": {tube: dataset.config[tube] for tube in tubes},
        "groups": groups,
        "pad_width": 2,
        "mapping": mapping,
        "cost_matrix": None,
        "train_epochs": 20,
    })

    model = train_som_classifier(train_dataset, validate_dataset, config)

    model.save(OUTPUT)
    model.save_information(OUTPUT)
