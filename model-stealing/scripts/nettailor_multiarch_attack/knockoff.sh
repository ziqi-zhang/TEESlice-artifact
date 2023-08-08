export PYTHONPATH=../:$PYTHONPATH

PROXY_MODEL=resnet18


for VIC_MODEL in resnet18 
do
for VIC_DATASET in CIFAR100
do
for ATT_MODEL in alexnet vgg16_bn vgg19_bn resnet18 resnet34 resnet50
do

python knockoff/adversary/train_nettailor.py \
models_nettailor/adversary_multiarch_attack/victim[$VIC_DATASET-$VIC_MODEL]-proxy[$PROXY_MODEL]-att[$ATT_MODEL]random \
$ATT_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 50,100,300,500,1000,3000,5000,10000,15000,20000,25000,30000 \
-d $1 \
--pretrained imagenet_for_cifar \
--log-interval 100 \
--epochs 10 \
--lr 0.1 \
--backbone-lr 0.01 \
--argmaxed \
&

done
done
done