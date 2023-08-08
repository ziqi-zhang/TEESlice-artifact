export PYTHONPATH=../:$PYTHONPATH


LR=0.01
for DATASET in CIFAR10 CIFAR100
do
for MODEL in resnet18 vgg16_bn alexnet resnet34 vgg19_bn
do

python knockoff/victim/train.py $DATASET $MODEL \
-d $1 \
-o models/victim/$DATASET-$MODEL \
-e 30 \
--log-interval 25 \
--pretrained imagenet_for_cifar \
--lr $LR \
--lr-step 20

done
done


LR=0.05
for DATASET in STL10
do
for MODEL in resnet18 vgg16_bn alexnet resnet34 vgg19_bn
do

python knockoff/victim/train.py $DATASET $MODEL \
-d $1 \
-o models/victim/$DATASET-$MODEL \
-e 30 \
--log-interval 25 \
--pretrained imagenet_for_face \
--lr $LR \
--lr-step 20 

done
done


LR=0.01
for DATASET in UTKFaceRace
do
for MODEL in resnet18 vgg16_bn alexnet resnet34 vgg19_bn
do

python knockoff/victim/train.py $DATASET $MODEL \
-d $1 \
-o models/victim/$DATASET-$MODEL \
-e 30 \
--log-interval 25 \
--pretrained imagenet_for_face \
--lr $LR \
--lr-step 20 

done 
done
