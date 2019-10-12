export CUDA_VISIBLE_DEVICES=0,1

python -u main.py --arch mobilenetv2 \
					   --depth 17 \
					   --batch-size 64 \
					   --no-tricks \
					   --rew \
					   --sparsity-type irregular \
					   --epoch 300 \
					   --optmzr adam \
					   --lr 0.001 \
					   --lr-scheduler cosine \
					   --warmup \
					   --warmup-epochs 5 \
					   --mixup \
					   --alpha 0.3 \
					   --smooth \
					   --smooth-eps 0.1 &&
echo "Congratus! Finished rew training!
"&&
python -u main.py --arch mobilenetv2 \
					   --depth 17 \
					   --batch-size 64 \
					   --no-tricks \
					   --masked-retrain \
					   --sparsity-type threshold \
					   --epoch 300 \
					   --optmzr adam \
					   --lr 0.001 \
					   --lr-scheduler cosine \
					   --warmup \
					   --warmup-epochs 5 \
					   --mixup \
					   --alpha 0.3 \
					   --smooth \
					   --smooth-eps 0.1 \
					   --config-file config_mobile_v2_0.7 &&
echo "Congratus! Finished retrain with config_mobilenet! No Tricks applied"
