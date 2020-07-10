GPUS=0

CUDA_VISIBLE_DEVICES=$GPUS python3 main.py  \
	--arch resnet152  \
  --batch_size 64