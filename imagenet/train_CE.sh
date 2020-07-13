MODEL='efficientnet-b0'
BATCH_SIZE=64
GPUS=13

echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 main.py /data/Imagenet      \
        --arch $MODEL                                           \
        --workers 4				                                    	\
        --T 3                                                   \
        --batch-size $BATCH_SIZE                                \
        --lr 6e-4                                               \
        --save_path "weights/${TEACHER}_${MODEL}" --epochs 300

echo "test done: $(date)"
