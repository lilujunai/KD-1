MODEL='efficientnet-b0'

GPUS=16,17,18,19

echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --save_path "weights/${TEACHER}_${MODEL}" --epochs 300

echo "test done: $(date)"
