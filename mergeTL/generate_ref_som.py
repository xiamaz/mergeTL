"""
Generate ref SOM for Merged FCS.
marker_name_only
"""
from collections import Counter

from flowcat import io_functions, utils
from flowcat.sommodels import casesom as fc_som
from flowcat.sommodels.fcssom import FCSSom
from flowcat.types.marker import Marker
from flowcat.constants import DEFAULT_REFERENCE_SOM_ARGS
from flowcat.utils.logs import setup_logging


def get_tube_marker(cases: "Iterable[Case]") -> "Dict[(Marker, str), float]":
    """Get markers in the given tube among cases."""
    samples = [s for c in cases for s in c.samples]
    markers = list({Marker(antibody=marker.antibody, color=None) for s in samples for marker in s.get_data().channels})
    markers_dict = {"1": markers}

    return markers_dict


def read_sel_markers(selected_markers) -> "Dict[(Marker, str), float]":
    """read selcted markers from a file and convert to Marker object"""
    markers = list(selected_markers.values())[0]
    marker_names = []
    for marker in markers:
        marker_names.append(Marker(antibody=Marker.name_to_marker(marker).antibody, color=None))
    selected_markers = {"1": marker_names}
    return selected_markers


dataset = io_functions.load_case_collection(
    utils.URLPath("/data/flowcat-data/2020_Nov_rerun/Merged_Files/MLL9F"),
    utils.URLPath("/data/flowcat-data/2020_Nov_rerun/Merged_Files/MLL9F_meta/train.json.gz"))

references = io_functions.load_json(
    utils.URLPath("/data/flowcat-data/2020_Nov_rerun/Merged_Files/MLL9F_meta/references.json"))


OUTPUT = utils.URLPath("/data/flowcat-data/2020_Nov_rerun/Merged_SOM/MLL9F")

setup_logging(None, "generate ref SOM for merged FCS")

ref_dataset = dataset.filter(labels=references)

tensorboard_dir = None

# Discover channels in the given dataset
markers = get_tube_marker(ref_dataset)
# markers = read_sel_markers(sel_markers)
print(markers)

io_functions.save_json(markers, OUTPUT / "markers.json")

# set marker_name_only to True
ref_config = DEFAULT_REFERENCE_SOM_ARGS.copy()
ref_config.marker_name_only = True

model = fc_som.CaseSom(
    tubes=markers,
    tensorboard_dir=tensorboard_dir,
    modelargs=ref_config,
)
model.train(ref_dataset)
io_functions.save_casesom(model, OUTPUT / "sommodel")
