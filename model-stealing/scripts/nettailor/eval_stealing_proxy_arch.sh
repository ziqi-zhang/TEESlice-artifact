export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
PROXY_MODEL=resnet18
for VIC_MODEL in alexnet
do
for VIC_DATASET in CIFAR100
do


python knockoff/adversary/eval_stealing_nettailor.py \
models_nettailor/adversary/victim[$VIC_DATASET-$VIC_MODEL]-proxy[$PROXY_MODEL]-random \
$PROXY_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 50,100 \
-d $1 \
# &


done
done