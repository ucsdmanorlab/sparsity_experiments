# Sparsity Experiments

This repository contains the experimental code used for the preprint **[A deep learning-based strategy for producing dense 3D segmentations from sparsely annotated 2D images](https://www.biorxiv.org/content/10.1101/2024.06.14.599135v1)**.

## Important Notice
**This repository is no longer being maintained.** It serves as an archive of the models and scripts used in the experiments conducted for the paper. For the most up-to-date implementation of these methods, please refer to https://github.com/ucsdmanorlab/bootstrapper.git

The code assumes some familiarity with gunpowder for the training and prediction pipelines. See [lsd](https://github.com/funkelab/lsd) and [gunpowder](https://github.com/funkelab/gunpowder) for helpful introductory tutorials and notebooks!

## Setup
```
git clone https://github.com/ucsdmanorlab/sparsity_experiments.git
cd sparsity_experiments
sh setup.sh
```
Requires Conda to be installed. 

This clones the repo and creates a conda environment named `sparsity` with all legacy dependencies. See [setup.sh](setup.sh) for more details.

## Directory Structure

The repository is organized as follows:
- [`example/data`](example/data): Includes scripts to 
    * [download](example/data/download.py) example 3D FIB-25 data
    * [convert](example/data/tif_labels_to_zarr.py) manually annotated labels to 3D zarr format
    * [make masks](example/data/make_unlabelled_mask.py) for training with sparse labels
    * [make 2D zarrs](example/data/make_2d.py) from 3D zarrs
    * [create scale pyramid](example/data/scale_pyramid.py), [partition labels](example/data/partition_labels.py), and [view stacked 2D zarr datasets](example/data/view_stacked.py)
    * [stack 2D predictions](example/data/stack_2d.py)

- [`example/sparse_nets`](example/sparse_nets/): Contains example models to learn dense [affinities](`example/sparse_nets/affs`), [LSDs](`example/sparse_nets/lsd`), or [both](`example/sparse_nets/mtlsd`) from sparse training data.
    * each target contains two example model directories: `2d_dense` and `3d_dense`, which are 2D U-Nets and 3D U-Nets, respectively. 
    * every model directory contains the following: 
        * `config.json`: model and training parameters
        * `train.py`: training script
        * `predict.py`: prediction script
        * `model.py`: model and loss definition
        * `unet.py`: U-Net module
        * `view.py`: visualization script for snapshots using neuroglancer
    * each sparse model's training script (`train.py`) automatically finds or prepares the appropriate training data using the name of the model directory.
        - "dense" models use all available labels
        - "disk_N" models use N disk-shaped masks to mask out labels outside the disks
        - custom sparsities can be used by preparing a zarr group in the training zarr which contains three datasets: `raw`, `labels`, and `unlabelled` which are the raw data, labels, and mask data, respectively. For example, `train.zarr/name_of_custom_sparsity/{raw,labels,unlabelled}`
- [`example/affs_nets`](example/affs_nets): Contains example models to learn 3D affinities from [stacked 2D affinities](`example/affs_nets/affs`), [stacked 2D LSDs](`example/affs_nets/lsds`), or [3D LSDs](`example/affs_nets/3d_lsds`) using synthetic 3D labels.
    * each input contains four variations in the synthhetic labels generation: `A`, `B`, `C`, and `Z`. 
    * individual model directories are structured similarly to the sparse models.
- [`example/bootstrapped_nets`](example/bootstrapped_nets): Contains one example 3D model to learn 3D affinities and 3D LSDs from dense pseudo ground truth training data. 
    * Model directory is the same as `sparse_nets/mtlsd/3d_dense`
    * Does not contain sparsity directory logic, instead contains logic to obtain training labels from a segmentation. 
- [`scripts`](scripts): Contains post-processing, evaluation, and visualization scripts.
    * watershed and hierarchical region agglomeration: [`scripts/hierarchical.py`](scripts/hierarchical.py)
    * evaluation: [`scripts/evaluate.py`](scripts/evaluate.py) -- uses [funlib.evaluate](https://github.com/funkelab/funlib.evaluate) to compute voxel-based and skeleton-based metrics

## Workflow

The general order of operations for bootstrapping 3D segmentations from sparse 2D labels is as follows:
1. Prepare 3D Zarr
    - Use the scripts in  `example/data` to prepare a 3D and 2D zarr container with the following datasets: `raw`, `labels`, and `unlabelled`. 

2. Train sparse model
    - Start training: Run `python train.py` in the chosen model directory
    - View snapshots: Use `python view.py -i path/to/snapshot.zarr` to see training snapshots

3. Predict with sparse model
    - Run prediction on all sections: Execute `python predict.py`. Use the 2D zarr container's raw datasets as inputs.

4. Stack 2D predictions
    - Use `example/data/stack_2d.py` to make a 3D array of stacked 2D predictions.

5. Train stacked 2D -> 3D affinities model
    - Go to `example/affs_nets` directory
    - Choose the model based on output of sparse model (e.g., affs, lsds, 3d_lsds)
    - Run `python train.py` in the selected model directory

6. Predict with affinities model
    - Generate 3D affinities: Run `python predict.py` with the stacked 2D predictions as input.

7. Segment 3D affinities
    - Use your preferred post-processing method on the 3D affinities to obtain a 3D segmentation.
    - For example, you can use hierarchical region agglomeration with `python scripts/hierarchical.py path/to/container.zarr name/of/3d_affs_dataset path/to/output.zarr name/of/3d_segmentation_dataset`

8. Train bootstrapped model
    - Execute `python train.py` in the boostrapped model directory. Ensure the paths in the training script are correct and point to the correct training data.

9. Final prediction and segmentation
    - Predict on full volume: Run `python predict.py`. Use the 3D zarr container's raw dataset as input.
    - Segment similarly to step 7.

Remember to adjust file paths and parameters as needed for your specific dataset and requirements.

## Citation

If you use this code or data in your research, please cite our preprint:
```
@article {Thiyagarajan2024.06.14.599135,
	author = {Thiyagarajan, Vijay Venu and Sheridan, Arlo and Harris, Kristen M. and Manor, Uri},
	title = {A deep learning-based strategy for producing dense 3D segmentations from sparsely annotated 2D images},
	year = {2024},
	doi = {10.1101/2024.06.14.599135},
	URL = {https://www.biorxiv.org/content/early/2024/06/15/2024.06.14.599135},
}
```

## Contact

For questions about the preprint or this repository, please contact vvenu@utexas.edu.

Remember, for the most current implementation of these methods, please use our [Bootstrapper](https://github.com/ucsdmanorlab/bootstrapper.git) tool.
