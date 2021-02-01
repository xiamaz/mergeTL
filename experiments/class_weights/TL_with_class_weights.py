"""
Transfer Learning exp1
1) groups - only CLL and normal
   model : base - 9F all CLL and normal samples
           target - 5F - increasing sample size (start with very few samples)

2) groups - only rare subtypes ( no CLL, MBl, normal)
   model : base - 9F all rare subtypes samples
           target - 5F - increasing sample size (start with very few samples)
"""

from flowcat import classifier, utils, io_functions
from flowcat.constants import DEFAULT_CLASSIFIER_CONFIG, GROUPS, DEFAULT_CLASSIFIER_ARGS
from flowcat import flowcat_api as fc_api
from flowcat.classifier import som_dataset
from flowcat.classifier.models import create_model_multi_input

import numpy as np
import math

def create_class_weight(labels_dict, mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

SOM_DATASET = utils.URLPath("/data/flowcat-data/2020-04_merged_train/MLL5F")
OUTPUT = utils.URLPath("/data/flowcat-data/2020-04_merged_train/TL/class_weights/mu_30/model_30")


def create_class_weight(labels_dict, mu=0.30):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight


LOGGER = utils.logs.setup_logging(None, "classify")

groups = GROUPS
tubes = ("1")
#tubes = ("1", "2")

mapping = None
dataset = som_dataset.SOMDataset.from_path(SOM_DATASET)

train_dataset, validate_dataset = fc_api.prepare_classifier_train_dataset(dataset, split_ratio=0.3, groups= groups, mapping= mapping, balance=None)
labels_dict = train_dataset.group_count

config = classifier.SOMClassifierConfig(**{
"tubes": {tube: dataset.config[tube] for tube in tubes},
"groups": groups,
"pad_width": 2,
"mapping": mapping,
"cost_matrix": None,
})

class_weight = create_class_weight(labels_dict)
#class_weight = utils.classification_utils.calculate_group_weights(labels_dict)
class_weight = {
    i: class_weight.get(g, 1.0) for i, g in enumerate(groups)
    }
print(class_weight)
model = fc_api.train_som_classifier(train_dataset, validate_dataset, config, class_weights= class_weight)

model.save(OUTPUT)
model.save_information(OUTPUT)

