# Evaluation
These instructions outline the pipeline for evaluating the Tag Map localizations against the coarse-localization metrics P2E and E2P as described in the paper. 

Currently the evaluation is only supported for the Matterport3D (MP3D) dataset which can be downloaded following the instructions [here](https://niessner.github.io/Matterport/).


## Setup
The evaluation assumes that the MP3D data folder has the following structure:
```
<mp3d_dir>
├── <scene 1>
│   ├── undistorted_color_images
│   ├── undistorted_depth_images
│   ├── matterport_camera_poses
│   ├── matterport_camera_intrinsics
│   ├── house_segmentations
|   └── ...
├── <scene 2>
│   ├── undistorted_color_images
│   ├── undistorted_depth_images
│   ├── matterport_camera_poses
│   ├── matterport_camera_intrinsics
│   ├── house_segmentations
|   └── ...
└── ...
```


## 1. Generate Tag Maps for all scenes
The Tag Maps for all MP3D scenes can be generated using:

```
python scripts/generate_tag_maps_matterport.py \
  --params config/tag_map_creation//matterport_ram.yaml \
  --output_dir <path_to_output_directory> \
  --matterport_dir <path_to_mp3d_dir>
```

Alternatively, pre-generated Tag Maps can be downloaded [here](https://huggingface.co/datasets/frozendonuts/tag-mapping/resolve/main/mp3d_tag_maps.zip). Please read and agree to the [MP3D EULA](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) before downloading.



## 2. Generate scene lattice graphs
Computing the coarse-localization metrics P2E and E2P requires computing the shortest paths between points in the scene. The shortest path computation is approximated using a lattice graph which spans the scene's free space while avoiding collisions with the scene geometry. Shortests paths are then computed and stored for each pair of nodes in the lattice graph.

The lattice graphs and precomputed shortest paths for all MP3D scenes can be generated using:
```
python scripts/generate_lattice_graph_matterport.py \
  --params config/lattice_graph_creation/matterport.yaml \
  --output_dir <path_to_output_directory> \
  --matterport_dir <path_to_mp3d_dir>
```

Alternatively, pre-generated lattice graphs can be downloaded [here](https://huggingface.co/datasets/frozendonuts/tag-mapping/resolve/main/mp3d_lattice_graphs.zip) (61 GB). Please read and agree to the [MP3D EULA](https://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf) before downloading.

Lattice graphs can be visualized using the included script:
```
python scripts/visualize_lattice_graph_matterport.py \
  --lattice_graph_path <path_to_lattice_graph_file> \
  --matterport_dir <path_to_mp3d_dir>
```


## 3. Run the evaluation
The evaluation is ran with the following command:
```
python scripts/evaluate_localization_matterport.py \
  --params <path_to_param_file> \
  --tag_maps_dir <path_to_tag_maps_directory> \
  --lattice_graphs_dir <path_to_lattice_graphs_directory> \
  --output_dir <path_to_output_directory> \
  --matterport_dir <path_to_mp3d_dir>
```

Evaluations are done separately for the labeled objects and labeled regions/locations depending on the setting of the params file. For running the object and region evaluations the param files `config/evaluation/matterport_objects.yaml` and `config/evaluation/matterport_regions.yaml` can be used respectively.

For each scene, the evaluation outputs are saved as a pickled Python dictionary.


## 4. Visualizing evaluation results
The evaluation saves an output file for every scene in the dataset. A notebook for visualizing the evaluation outputs for a scene can be found under the `notebooks` folder. 
