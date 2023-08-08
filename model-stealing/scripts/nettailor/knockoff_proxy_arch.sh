export PYTHONPATH=../:$PYTHONPATH

PROXY_MODEL=resnet18


for VIC_MODEL in alexnet
do
for VIC_DATASET in CIFAR10 
do

python knockoff/adversary/train_nettailor.py \
models_nettailor/adversary/victim[$VIC_DATASET-$VIC_MODEL]-proxy[$PROXY_MODEL]-random \
$PROXY_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 50,100 \
-d $1 \
--pretrained imagenet_for_cifar \
--log-interval 100 \
--epochs 10 \
--lr 0.1 \
--backbone-lr 0.01 \
--argmaxed \
# &

done
done