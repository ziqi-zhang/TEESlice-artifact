export PYTHONPATH=../:$PYTHONPATH

MODEL=resnet18
for DATASET in CIFAR10 CIFAR100
do

python mem_attack.py $DATASET $MODEL \
-d $1 \
-o results/membership/$DATASET-$MODEL \
--victim_dir results/victim/$DATASET-$MODEL \
-e 100 \
--lr 1e-2 \
--pretrained &



done