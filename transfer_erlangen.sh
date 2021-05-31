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
OUTPUT=/ceph01/projects/flowcat/output/05-21_convergence
SOM_ROOT=/ceph01/projects/flowcat/output/merge_TL_Nov_run/Merged_SOM
SOM_DATASET=Erlangen
SOM_OUTPUT="$SOM_ROOT/${SOM_DATASET}"
MODEL_PANEL=Other
MODEL_ROOT=/ceph01/projects/flowcat/output/merge_TL_Nov_run/Merged_model/base_models
BASE_MODEL=MLL9F_Bonn_Berlin
BASE_PATH="$MODEL_ROOT/$BASE_MODEL"
EPOCHS=50
REP=$2
MODEL_OUTPUT="$OUTPUT/transfer_${SOM_DATASET}_ep${EPOCHS}_${REP}"
#################

# I. Create reference SOM
echo "Output to $MODEL_OUTPUT"
if [ ! -d $MODEL_OUTPUT ]; then
    python3.6 mergeTL/merge_TL.py $SOM_OUTPUT $MODEL_OUTPUT $MODEL_PANEL $BASE_PATH $EPOCHS
else
    echo "Trained model found at $MODEL_OUTPUT. Skipping..."
fi
