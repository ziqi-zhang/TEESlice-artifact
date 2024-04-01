export PYTHONPATH=../:$PYTHONPATH

MODEL=alexnet
for MODEL in vgg19_bn 
do
for DATASET in CIFAR10 CIFAR100 STL10 UTKFaceRace
do

python train_victim.py $DATASET $MODEL \
-d $1 \
-o results/victim/$DATASET-$MODEL \
-e 100 \
--lr 1e-2 \
--pretrained 

done
done
