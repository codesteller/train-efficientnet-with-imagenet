# EfficientNet For TensorFlow 2.4

This repository provides a script and recipe to train the EfficientNet model on Large Scale Imagenet Dataset for varios EfficientNet architectures.
The content of the repository is tested by NVIDIA and part of the repository [Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)

## Table Of Contents

- [Setup](#setup)
    * [Requirements](#requirements)
- [Quick Start Guide](#quick-start-guide)
- [Advanced](#advanced)
    * [Scripts and sample code](#scripts-and-sample-code)
    * [Parameters](#parameters)
    * [Command-line options](#command-line-options)
    * [Getting the data](#getting-the-data)
    * [Training process](#training-process)
    * [Inference process](#inference-process)
- [Release notes](#release-notes)
    * [Changelog](#changelog)
    * [Known issues](#known-issues)

## Setup

The following section lists the requirements that you need to meet in order to start training the EfficientNet model.

### Requirements

This repository contains Dockerfile which extends the TensorFlow NGC container and encapsulates some dependencies. Aside from these dependencies, ensure you have the following components:
-   [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
-   [TensorFlow 20.08-py3] NGC container or later
-   Supported GPUs:
    - [NVIDIA Volta architecture](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/)
    - [NVIDIA Turing architecture](https://www.nvidia.com/en-us/geforce/turing/)
    - [NVIDIA Ampere architecture](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/)

For more information about how to get started with NGC containers, see the following sections from the NVIDIA GPU Cloud Documentation and the Deep Learning Documentation:
-   [Getting Started Using NVIDIA GPU Cloud](https://docs.nvidia.com/ngc/ngc-getting-started-guide/index.html)
-   [Accessing And Pulling From The NGC Container Registry](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#accessing_registry)
-   [Running TensorFlow](https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/running.html#running)
  
As an alternative  to the use of the Tensorflow2 NGC container, to set up the required environment or create your own container, see the versioned [NVIDIA Container Support Matrix](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).

For multi-node, the sample provided in this repository requires [Enroot](https://github.com/NVIDIA/enroot) and [Pyxis](https://github.com/NVIDIA/pyxis) set up on a [SLURM](https://slurm.schedmd.com) cluster.

## Quick Start Guide

To train your model using mixed or TF32 precision with Tensor Cores or using FP32, perform the following steps using the default parameters of the EfficientNet model on the ImageNet dataset. For the specifics concerning training and inference, see the [Advanced](#advanced) section.

1. Clone the repository.

	```
    git clone https://github.com/codesteller/train-efficientnet-with-imagenet.git efficientnet-train
    
    cd efficientnet-train
	```

2. Download and prepare the dataset.
    * Download ImageNet Dataset from the [link](https://www.image-net.org/download.php) to a folder "~/Downloads/imagenet12"
        ``` 
        $ ls -l ~/Downloads/imagenet12
        ILSVRC2012_img_test_v10102019.tar
        ILSVRC2012_img_train_t3.tar
        ILSVRC2012_img_train.tar
        ILSVRC2012_img_val.tar
        ```

    * To prepare the dataset 
        1. Run Docker Container
        ```
            docker run -it --rm --shm-size=1g --ipc=host \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v ~/Downloads/imagenet12:/imagenet \
            -w /workspace/nvidia-examples/build_imagenet_data/ \
            nvcr.io/nvidia/tensorflow:21.07-tf2-py3
        ```
        2. Process and Prepare Imagenet Dataset
        ```
            mkdir -p /imagenet/raw-data
            cp /imagenet/ILSVRC2012_img_*.tar /imagenet/raw-data/
            ./download_and_preprocess_imagenet.sh /imagenet
        ```

3. Start EfficientNet Docker built on top of the NGC container for training. Configure the dataset path accordingly
           ```
           bash startDocker.sh
           ```

4. Run training - change parameters in the file `train_efficientnet.sh` appropriately file the architecture, the dataset path. 
           ```
           bash ./scripts/docker/launch.sh
           ```

5. Start training.


## Advanced

The following sections provide greater details of the dataset and running training

### Parameters

Important parameters for training are listed below with default values.

- `mode` (`train_and_eval`,`train`,`eval`,`prediction`) - the default is `train_and_eval`.
- `arch` - the default is `efficientnet-b0`
- `model_dir` - The folder where model checkpoints are saved (the default is `/workspace/output`)
- `data_dir` - The folder where data resides (the default is `/data/`)
- `augmenter_name` - Type of Augmentation (the default is `autoaugment`)
- `max_epochs` - The number of training epochs (the default is `300`)
- `warmup_epochs` - The number of epochs of warmup (the default is `5`)
- `train_batch_size` - The training batch size per GPU (the default is `32`)
- `eval_batch_size` - The evaluation batch size per GPU (the default is `32`)
- `lr_init` - The learning rate for a batch size of 128, effective learning rate will be automatically scaled according to the global training batch size (the default is `0.008`)

The main script `main.py` specific parameters are:
```
 --model_dir MODEL_DIR
                        The directory where the model and training/evaluation
                        summariesare stored.
  --save_checkpoint_freq SAVE_CHECKPOINT_FREQ
                        Number of epochs to save checkpoint.
  --data_dir DATA_DIR   The location of the input data. Files should be named
                        `train-*` and `validation-*`.
  --mode MODE           Mode to run: `train`, `eval`, `train_and_eval`, `predict` or
                        `export`.
  --arch ARCH           The type of the model, e.g. EfficientNet, etc.
  --dataset DATASET     The name of the dataset, e.g. ImageNet, etc.
  --log_steps LOG_STEPS
                        The interval of steps between logging of batch level
                        stats.
  --use_xla             Set to True to enable XLA
  --use_amp             Set to True to enable AMP
  --num_classes NUM_CLASSES
                        Number of classes to train on.
  --batch_norm BATCH_NORM
                        Type of Batch norm used.
  --activation ACTIVATION
                        Type of activation to be used.
  --optimizer OPTIMIZER
                        Optimizer to be used.
  --moving_average_decay MOVING_AVERAGE_DECAY
                        The value of moving average.
  --label_smoothing LABEL_SMOOTHING
                        The value of label smoothing.
  --max_epochs MAX_EPOCHS
                        Number of epochs to train.
  --num_epochs_between_eval NUM_EPOCHS_BETWEEN_EVAL
                        Eval after how many steps of training.
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of steps of training.
  --warmup_epochs WARMUP_EPOCHS
                        Number of steps considered as warmup and not taken
                        into account for performance measurements.
  --lr_init LR_INIT     Initial value for the learning rate.
  --lr_decay LR_DECAY   Type of LR Decay.
  --lr_decay_rate LR_DECAY_RATE
                        LR Decay rate.
  --lr_decay_epochs LR_DECAY_EPOCHS
                        LR Decay epoch.
  --weight_decay WEIGHT_DECAY
                        Weight Decay scale factor.
  --weight_init {fan_in,fan_out}
                        Model weight initialization method.
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size per GPU.
  --augmenter_name AUGMENTER_NAME
                        Type of Augmentation during preprocessing only during
                        training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Evaluation batch size per GPU.
  --resume_checkpoint   Resume from a checkpoint in the model_dir.
  --use_dali            Use dali for data loading and preprocessing of train
                        dataset.
  --use_dali_eval       Use dali for data loading and preprocessing of eval
                        dataset.
  --dtype DTYPE         Only permitted
                        `float32`,`bfloat16`,`float16`,`fp32`,`bf16`
```

### Command-line options

To see the full list of available options and their descriptions, use the `-h` or `--help` command-line option, for example:
`python main.py --help`


### Getting the data

Refer to the [TFDS ImageNet readme](https://github.com/tensorflow/datasets/blob/master/docs/catalog/imagenet2012.md) for manual download instructions.
To train on ImageNet dataset, pass `$path_to_ImageNet_tfrecords` to `$data_dir` in the command-line.

Name the TFRecords in the following scheme:

- Training images - `/data/train-*`
- Validation images - `/data/validation-*`

### Training process

The training process can start from scratch, or resume from a checkpoint.

By default, bash script `scripts/{B0, B4}/training/{AMP, FP32, TF32}/convergence_8x{A100-80G, V100-16G, V100-32G}.sh` will start the training process from scratch with the following settings.
   - Use 8 GPUs by Horovod
   - Has XLA enabled
   - Saves checkpoints after every 5 epochs to `/workspace/output/` folder
   - AMP or FP32 or TF32 based on the folder `scripts/{B0, B4}/training/{AMP, FP32, TF32}`

To resume from a checkpoint, include `--resume_checkpoint` in the command-line and place the checkpoint into `--model_dir`.

#### Multi-node

Multi-node runs can be launched on a Pyxis/enroot Slurm cluster (see [Requirements](#requirements)) with the `run_{B0, B4}_multinode.sub` script with the following command for a 4-node NVIDIA DGX A100 example:

```
PARTITION=<partition_name> sbatch N 4 --ntasks-per-node=8 run_B0_multinode.sub
PARTITION=<partition_name> sbatch N 4 --ntasks-per-node=8 run_B4_multinode.sub
```
 
Checkpoint after `--save_checkpoint_freq` epochs will be saved in `checkpointdir`. The checkpoint will be automatically picked up to resume training in case it needs to be resumed. Cluster partition name has to be provided `<partition_name>`.
 
Note that the `run_{B0, B4}_multinode.sub` script is a starting point that has to be adapted depending on the environment. In particular, variables such as `--container-image` handle the container image to train using and `--datadir` handle the location of the ImageNet data.
 
Refer to the files contents to see the full list of variables to adjust for your system.

### Inference process

Validation is done every epoch and can be also run separately on a checkpointed model.

`bash ./scripts/{B0, B4}/evaluation/evaluation_{AMP, FP32, TF32}_8x{A100-80G, V100-16G, V100-32G}.sh`

Metrics gathered through this process are as follows:

```
- eval_loss
- eval_accuracy_top_1
- eval_accuracy_top_5
- avg_exp_per_second_eval
- avg_exp_per_second_eval_per_GPU
- avg_time_per_exp_eval : Average Latency
- latency_90pct : 90% Latency
- latency_95pct : 95% Latency
- latency_99pct : 99% Latency
```

To run inference on a JPEG image, you have to first store the checkpoint in the `--model_dir` and store the JPEG images in the following directory structure:

    ```
    infer_data
    |   ├── images
    |   |   ├── image1.JPEG
    |   |   ├── image2.JPEG
    ```

Run: 
`bash ./scripts/{B0, B4}/inference/inference_{AMP, FP32, TF32}.sh`

## Performance

The performance measurements in this document were conducted at the time of publication and may not reflect the performance achieved from NVIDIA’s latest software release. For the most up-to-date performance measurements, go to [NVIDIA Data Center Deep Learning Product Performance](https://developer.nvidia.com/deep-learning-performance-training-inference).

### Benchmarking

The following section shows how to run benchmarks measuring the model performance in training and inference modes.

#### Training performance benchmark

Training benchmark for EfficientNet-B0 was run on NVIDIA DGX A100 and NVIDIA DGX-1 V100 16GB.

To benchmark training performance with other parameters, run:

`bash ./scripts/B0/training/{AMP, FP32, TF32}/train_benchmark_8x{A100-80G, V100-16G}.sh`

Training benchmark for EfficientNet-B4 was run on NVIDIA DGX A100 and NVIDIA DGX-1 V100 32GB.

`bash ./scripts/B4/training/{AMP, FP32, TF32}/train_benchmark_8x{A100-80G, V100-16G}.sh`




## Release notes

### Changelog

March 2021
- Initial release
July 2021 
- Repository Forked and customized

### Known issues

- EfficientNet-B0 does not improve training speed by using AMP as compared to FP32, because of the CPU bound Auto-augmentation.


