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


#MARKERS = io_functions.load_json(utils.URLPath("/data/flowcat-data/2020-04_merged_train/MLL9F/markers.json"))
SOM_DATASET = utils.URLPath("/data/flowcat-data/2020-04_merged_train/MLL9F")
OUTPUT = utils.URLPath("/data/flowcat-data/2020-04_merged_train/MLL9F/Exp1")

LOGGER = utils.logs.setup_logging(None, "classify")

groups = [ "MCL", "PL", "LPL", "MZL", "FL", "HCL"] 
tubes = ("1")

mapping = None
dataset = som_dataset.SOMDataset.from_path(SOM_DATASET)
train_dataset, validate_dataset = fc_api.prepare_classifier_train_dataset(dataset, split_ratio=0.9, groups=groups, mapping=mapping, balance=None)

config = classifier.SOMClassifierConfig(**{
"tubes": {tube: dataset.config[tube] for tube in tubes},
"groups": groups,
"pad_width": 2,
"mapping": mapping,
"cost_matrix": None,
})
model = fc_api.train_som_classifier(train_dataset, validate_dataset, config)

model.save(OUTPUT)
model.save_information(OUTPUT)

