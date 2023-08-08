export PYTHONPATH=../:$PYTHONPATH

INCLUDE_NORM=1
INCLUDE_FC=1

ORI_DATASET=CIFAR10
TRAINED_DATASET=CIFAR10
MODEL_ARCH=resnet18
LR=0.001

for MODEL_NAME in resnet18 vgg16_bn alexnet resnet34 vgg19_bn
do

for THETA in 1.0 0.9 0.8 0.7 0.5 0.4 0.3 0.2 0.1 0.0
do

MODEL_ARCH=$MODEL_NAME
OUT_DIR=models/soter-var/$TRAINED_DATASET-$MODEL_NAME-theta$THETA-recover

python knockoff/attack/soter_recover_scalar.py $MODEL_NAME $MODEL_ARCH $THETA $OUT_DIR \
--ori_dataset $ORI_DATASET \
--pretrained imagenet_for_cifar \
--trained_dataset $TRAINED_DATASET \
--trained_ckpt_path models/victim/$TRAINED_DATASET-$MODEL_NAME/checkpoint.pth.tar \
-d $1 \
--img_dir pretrained_images \
--transferset_dir models/adversary/victim[$TRAINED_DATASET-$MODEL_NAME]-random-grey \
--budgets 50 \
--epochs 10 \
--log-interval 25 \
--lr $LR \
--include_norm $INCLUDE_NORM \
--include_fc $INCLUDE_FC \

done

done


ORI_DATASET=CIFAR100
TRAINED_DATASET=CIFAR100
MODEL_ARCH=resnet18
LR=0.001

for MODEL_NAME in resnet18 vgg16_bn alexnet resnet34 vgg19_bn
do

for THETA in 1.0 0.9 0.8 0.7 0.5 0.4 0.3 0.2 0.1 0.0
do

MODEL_ARCH=$MODEL_NAME
OUT_DIR=models/soter-var/$TRAINED_DATASET-$MODEL_NAME-theta$THETA-recover

python knockoff/attack/soter_recover_scalar.py $MODEL_NAME $MODEL_ARCH $THETA $OUT_DIR \
--ori_dataset $ORI_DATASET \
--pretrained imagenet_for_cifar \
--trained_dataset $TRAINED_DATASET \
--trained_ckpt_path models/victim/$TRAINED_DATASET-$MODEL_NAME/checkpoint.pth.tar \
-d $1 \
--img_dir pretrained_images \
--transferset_dir models/adversary/victim[$TRAINED_DATASET-$MODEL_NAME]-random-grey \
--budgets 500 \
--epochs 10 \
--log-interval 25 \
--lr $LR \
--include_norm $INCLUDE_NORM \
--include_fc $INCLUDE_FC \

done
done


ORI_DATASET=UTKFaceRace
TRAINED_DATASET=UTKFaceRace
MODEL_ARCH=resnet18
LR=0.001

for MODEL_NAME in resnet18 vgg16_bn alexnet resnet34 vgg19_bn
do

for THETA in 1.0 0.9 0.8 0.7 0.5 0.4 0.3 0.2 0.1 0.0
do
MODEL_ARCH=$MODEL_NAME
OUT_DIR=models/soter-var/$TRAINED_DATASET-$MODEL_NAME-theta$THETA-recover

python knockoff/attack/soter_recover_scalar.py $MODEL_NAME $MODEL_ARCH $THETA $OUT_DIR \
--ori_dataset $ORI_DATASET \
--pretrained imagenet_for_face \
--trained_dataset $TRAINED_DATASET \
--trained_ckpt_path models/victim/$TRAINED_DATASET-$MODEL_NAME/checkpoint.pth.tar \
-d $1 \
--img_dir pretrained_images \
--transferset_dir models/adversary/victim[$TRAINED_DATASET-$MODEL_NAME]-random-grey \
--budgets 50 \
--epochs 10 \
--log-interval 25 \
--lr $LR \
--include_norm $INCLUDE_NORM \
--include_fc $INCLUDE_FC \

done
done


ORI_DATASET=STL10
TRAINED_DATASET=STL10
MODEL_ARCH=resnet18 vgg16_bn alexnet resnet34 vgg19_bn
LR=0.001

for MODEL_NAME in vgg19_bn
do

for THETA in 1.0 0.9 0.8 0.7 0.5 0.4 0.3 0.2 0.1 0.0
do

MODEL_ARCH=$MODEL_NAME
OUT_DIR=models/soter-var/$TRAINED_DATASET-$MODEL_NAME-theta$THETA-recover

python knockoff/attack/soter_recover_scalar.py $MODEL_NAME $MODEL_ARCH $THETA $OUT_DIR \
--ori_dataset $ORI_DATASET \
--pretrained imagenet_for_face \
--trained_dataset $TRAINED_DATASET \
--trained_ckpt_path models/victim/$TRAINED_DATASET-$MODEL_NAME/checkpoint.pth.tar \
-d $1 \
--img_dir pretrained_images \
--transferset_dir models/adversary/victim[$TRAINED_DATASET-$MODEL_NAME]-random-grey \
--budgets 50 \
--epochs 10 \
--log-interval 25 \
--lr $LR \
--use_STL10 \
--include_norm $INCLUDE_NORM \
--include_fc $INCLUDE_FC \

done

done
