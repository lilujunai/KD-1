MODEL='efficientnet-b0'
BATCH_SIZE=512
GPUS=13,14,15,16,17

echo "start: $(date)"
CUDA_VISIBLE_DEVICES=$GPUS python3 prune.py /data/Imagenet      \
        --arch $MODEL                                           \
        --workers 8						\
        --batch-size $BATCH_SIZE                                \
        --lr 6e-5                                               \
        --pth_path  "./weights/resnet152_efficientnet-b0/model_best:EfficientNet_ResNet.pth.tar" \
        --save_path "weights/${MODEL}_pruned" --epochs 300