export PYTHONPATH=../:$PYTHONPATH

MODEL=alexnet
for MODEL in  vgg19_bn
do
for DATASET in CIFAR100 
do
for MODE in  block_shallow
do

python train_layer.py $DATASET $MODEL \
-d $1 \
-o results/adversary/train_layer/$DATASET-$MODEL \
--victim_dir results/victim/$DATASET-$MODEL \
--shadow_model_dir results/adversary/shadow/$DATASET-$MODEL \
--budgets 100,300 \
--lr 0.1 \
--remain-lr 1e-3 \
--update-lr 1e-2 \
--epochs 10 \
--pretrained \
--graybox-mode $MODE \
--argmaxed 

done
done
done