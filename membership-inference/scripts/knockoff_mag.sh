export PYTHONPATH=../:$PYTHONPATH

MODEL=alexnet
for MODEL in   vgg16_bn vgg19_bn
do
for DATASET in CIFAR100
do
for MODE in block_large_mag
do

python train_mag.py $DATASET $MODEL \
-d $1 \
-o results/adversary/train_mag/$DATASET-$MODEL \
--victim_dir results/victim/$DATASET-$MODEL \
--shadow_model_dir results/adversary/shadow/$DATASET-$MODEL \
--budgets 100,300 \
--lr 0.01 \
--epochs 10 \
--pretrained \
--graybox-mode $MODE \
--argmaxed &

done
done
done