export PYTHONPATH=../:$PYTHONPATH

PROXY_MODEL=resnet18
for VIC_DATASET in CIFAR100
do
for VIC_MODEL in resnet18
do
for ATT_MODEL in resnet18 resnet34 resnet50 alexnet vgg16_bn vgg19_bn
do

python knockoff/adversary/transfer.py random \
models/victim/$VIC_DATASET-$VIC_MODEL \
--queryset $VIC_DATASET \
--out_dir models_nettailor/adversary_multiarch_attack_baseline/victim[$VIC_DATASET-$VIC_MODEL]-proxy[$PROXY_MODEL]-att[$ATT_MODEL]random \
--budget 30000 \
--queryset $VIC_DATASET \
--batch_size 32 \
-d $1 \
&

done
done
done