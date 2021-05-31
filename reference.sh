set -e
set -u
# FlowCat results generation: Usage
#
# Necessary data: Main dataset including training and test data
# Additional unused data (AML, MM, HCLv data): used for generation of tSNE plots
#
# Change following paths as needed for your system. All generated data and plots
# will be saved in the given OUTPUT directory.

#################
# Configuration #
#################
DATA_ROOT=/ceph01/projects/flowcat/data/Original_FCS_files/MLL-flowdata
DATA=$DATA_ROOT/decCLL-9F
META=$DATA_ROOT/decCLL-9F.2019-10-29.meta/train.json.gz
META_TEST="$DATA_ROOT/decCLL-9F.2019-10-29.meta/test.json.gz"
LABELS=$DATA_ROOT/decCLL-9F.2019-10-29.meta/references.json

DATA_UNUSED=/data/flowcat-data/paper-cytometry/unused-data/data
META_UNUSED=/data/flowcat-data/paper-cytometry/unused-data/meta.json.gz

OUTPUT=/ceph01/projects/flowcat/output/05-21_convergence
SOM_ROOT=/ceph01/projects/flowcat/output/merge_TL_Nov_run/Merged_SOM
SOM_OUTPUT="$SOM_ROOT/MLL9F"
EPOCHS=100
REP=$2
MODEL_OUTPUT="$OUTPUT/classifier_ep${EPOCHS}_${REP}"
#################

# I. Create reference SOM
echo "Output to $MODEL_OUTPUT"
if [ ! -d $MODEL_OUTPUT ]; then
    python3.6 mergeTL/merged_model.py $SOM_OUTPUT $MODEL_OUTPUT MLL $EPOCHS
else
    echo "Trained model found at $MODEL_OUTPUT. Skipping..."
fi
