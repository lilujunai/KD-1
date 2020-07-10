GPUS=0,1

CUDA_VISIBLE_DEVICES=$GPUS python3 prune.py \
	--l2 1  \
	--dist 0.1  \
	--lr 1e-5 \
	--epochs 100  \
	--batch_size 64 \
	--pth_path './checkpoint/EfficientNet.pth'
