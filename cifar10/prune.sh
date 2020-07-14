GPUS=14

CUDA_VISIBLE_DEVICES=$GPUS python3 prune.py \
	--lr 1e-5 \
	--epochs 300  \
	--batch_size 64 
	
