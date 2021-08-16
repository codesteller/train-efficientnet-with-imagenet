docker run --gpus all -it --rm --net=host --ipc=host --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add DAC_READ_SEARCH --security-opt seccomp=unconfined \
-v $(pwd)/:/workspace/ \
-v "/home/atos1/pallab/workspace/cv_benchmarking/dataset/imagenet/":/data/ \
-v "/home/atos1/pallab/workspace/cv_benchmarking/dataset/imagenet/raw-data/imagenet_infer":/infer_data/images/ \
codesteller/efficientnet-tf2:21.07-tf2-py3