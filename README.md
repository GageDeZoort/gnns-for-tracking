# Graph Neural Networks for Charged Particle Tracking
This repository is focused on applying graph neural networks (GNNs) to the task of charged particle track reconstruction using the high-pileup TrackML dataset:
- TrackML @ Kaggle: https://www.kaggle.com/c/trackml-particle-identification
- TrackML @ Codalab: https://competitions.codalab.org/competitions/20112
TrackML data is a 3D point cloud of tracker hits with associated truth information about the particles that generate them. The goal of GNN-based tracking workflows is to embed track hits as graph nodes and apply GNNs to cluster hits belonging to the same particle. This repo focuses on two compelemtary strategies: edge classification to predict hit associations and object condensation to cluster hits and predict track properties. 

## Graph Construction 
Base directory: ```graph_construction/```. TrackML provides several truth quanities about the particles producing track hits in each events, for example transverse momentum, vertex, and charge. Each track hit is uniquely associated with a particle ID, so that we can calculate additional information about each track at truth level. 
- ```measure_particle_properties.py``` produces a dataframe of truth information corresponding to each particle, including transverse momentum, charge, transverse impact parameter, number of hits, number of layers hit, and whether the particle skips a layer. Particles that produce hits in three or more layers, do not skip a layer, and follow a physical trajectory are labeled as reconstructable. <br /><br />*Example usage*:<br />
 ```python measure_particle_properties.py -i /trackml_data/train_1 -o particle_properties --n-workers=3 ```
<br />

- ```slurm/measure_particle_properties.{py,slurm}``` are provided to produce particle dataframes as set of batch jobs via Slurm.

The following scripts build tracker hit graphs from a set of TrackML event files and corresponding particle property dataframes. 
- ```build_graphs.py``` produces graphs containing track hits embedded as nodes with features (r, phi, z, u, v), where u and v are coordinates in conformal space, and edge features (dr, dphi, dz, dR), where dR is the hit-hit distance in eta-phi space. Hits are assigned to particle IDs and the track parameters belonging to that particle ID at truth level. Edges are drawn via a set of geometric selections specified in ```configs/build_graphs.yaml```. <br /><br />*Example usage*:<br />
 ```python train_TCN.py -i graphs/train1_ptmin0p8/ --n-train=10000 --n-test=2000 --learning-rate=0.0001```
<br />

- ```slurm/build_graphs_job.{py,slurm}``` are provided to produce graphs via a set of batch jobs via Slurm.
- ```slurm/optimize_graph_construction_params.{py,slurm}``` are provided to submit batch jobs corresponding to different geometric selections and report the corresponding graph construction efficiency (n_true/n_true_possible edges) and purity (n_true/n_total edges).

The graph construction routine employs several functions available in ```utils/graph_building_utils.py``` and ```utils/hit_processing_utils.py``` (see below). 

## GNN Inference 
### Models
### Training
Two training scripts, ```train_IN.py``` and ```train_TCN.py```, are located at the head of the directory. ```train_IN.py``` focuses on training IN-based edge classification architectures with no learned track finding step. ```train_TCN.py``` focuses on the object condensation approach, optionally including track parameter predictions. Each script may be run stand-alone or through a batch job. <br /><br />*Example usage*:<br />
 ```python build_graphs.py /configs/build_graphs.yaml --n-workers=3 ```
<br />


Additionally, hyperparameter job arrays may be submitted using the scripts in ```hyperparameter_scans```. 
### Inference 

## Track-Finding 

## Utils
- ```hit_processing_utils.py``` contains a set of helper functions for opening TrackML events.
- ```graph_building_utils.py``` contains functions that select hits given a set of truth cuts, select edges given a set of geometric cuts, split the detector into multiple sectors, and correct edge truth levels in the case that multiple barrel-endcap layer connections are possible. 
- ```data_utils.py``` contains a set of functions for loading in graphs, building train/test/validation partitions, organizing graph datasets and dataloaders, and a custom GraphDataset extension of the PyTorch Geometric Dataset class. 
- ```inference_utils.py``` contains functions relevant to training and testing various GNN algorithms. 
