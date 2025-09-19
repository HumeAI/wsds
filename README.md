# WSDS

This is a library containing some general purpose classes and utilities for multimodal audio data management at Hume.

The goal of this library is to support :

- Inspecting the data
- Streaming/Batching samples from the data
- Querying/Filtering the data
- Saving out subsets of the data
- Using data via dataloaders in training

## Getting Started  

```bash
# create environment
conda create -n hume_wsds python=3.10
conda activate hume_wsds

# install hume_wsds
pip install -e .

# install GNU parallel
sudo apt-get install parallel
```
 
## Converting a Dataset

1. Adjust the dataset paths in config:  
   `examples/data_conversion/configs/template.yaml` 

2. Ensure that the keys for required segmentation are defined in `ShardMapping`
   `hume_wsds/constants.py`

   e.g., for a v4 pipeline, with conventional vad
    ```bash 
    "v4-vad_ws_mvad.raw.vad.npy": ["v4-vad_ws_mvad", "raw.vad.npy"],
    "v4-vad_ws_mvad.eq.vad.npy":  ["v4-vad_ws_mvad", "eq.vad.npy"],
    "v4-vad_ws_mvad.max.vad.npy": ["v4-vad_ws_mvad", "max.vad.npy"],
    ```
    
4. Run the conversion script:  
   `./examples/data_conversion/convert_dataset.sh`

   
## Viewing a converted dataset 

Examples for viewing and analyzing the converted datasets can be found in the `examples/` directory. 