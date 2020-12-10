# Merge TL: Scripts to generate trained models for merged datsets using flowcat api

## For each datset, generate 2 models: one without any knowledge transfer and a second one with knowledge transfer from the base model

### Scripts
1. **generate_ref_som.py:** generate reference SOM for merged base dataset(MLL9F). We will use this reference SOM for generating individual SOMs for each dataset.
2. **merged_model.py:** train merged model
3. **merge_TL.py:** train target models with knowledge transfer from the given base model. create target model, set weights from the base model.

### Excution
* ./generate_ref_som.py
	
* merged_model.py:
   * ./merged_model.py "SOM datapath" "outputpath" "panel"
 
* merge_TL.py:
   * ./merge_TL.py "SOM datapath" "outputpath" "panel" "basemodel path"

### Parameters
* SOM datapath : path to merged SOM files
* outputpath: Out put folder path to save model and meta information
* basemodel path: path where the base model is saved ( parent folfer containing the model.h5 file)
* panel: one of the following
	* Bonn, Berlin, MLL

### Dependencies:
* flowcat