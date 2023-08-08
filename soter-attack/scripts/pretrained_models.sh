export PYTHONPATH=../:$PYTHONPATH

DATASET=CIFAR10
MODEL_ARCH=resnet18
LR=0.01


for MODEL_NAME in resnet18 vgg16_bn alexnet
do
MODEL_ARCH=$MODEL_NAME
python knockoff/attack/pretrained_models.py $DATASET $MODEL_NAME $MODEL_ARCH \
-d $1 \
-o models/victim/$DATASET-$MODEL_NAME \
-e 30 \
--log-interval 25 \
--pretrained imagenet_for_cifar \
--lr $LR \
--lr-step 20 

done



# DATASET=UTKFaceRace
# MODEL_ARCH=resnet18
# LR=0.01

# for MODEL_NAME in resnet18 vgg16_bn alexnet
# do
# MODEL_ARCH=$MODEL_NAME
# python knockoff/attack/pretrained_models.py $DATASET $MODEL_NAME $MODEL_ARCH \
# -d $1 \
# -o models/victim/$DATASET-$MODEL_NAME \
# -e 30 \
# --log-interval 25 \
# --pretrained imagenet_for_face \
# --lr $LR \
# --lr-step 20 

# done


# for MODEL_NAME in gluon_resnet18_v1b resnet18 ssl_resnet18 swsl_resnet18 
# do
# python knockoff/attack/pretrained_models.py $DATASET $MODEL_NAME $MODEL_ARCH \
# -d $1 \
# -o models/victim/pretrained-$DATASET-timm-$MODEL_NAME \
# -e 30 \
# --log-interval 25 \
# --lr $LR \
# --lr-step 20 

# done


# for MODEL_NAME in FractalDB-1000_res18 FractalDB-10000_res18
# do
# python knockoff/attack/pretrained_models.py $DATASET $MODEL_NAME $MODEL_ARCH \
# -d $1 \
# -o models/victim/pretrained-$DATASET-$MODEL_NAME \
# --pretrained_dir downloaded_models \
# -e 30 \
# --log-interval 25 \
# --lr $LR \
# --lr-step 20 \

# done


# for MODEL_NAME in resnet18_places365 resnet18-kaggle
# do
# python knockoff/attack/pretrained_models.py $DATASET $MODEL_NAME $MODEL_ARCH \
# -d $1 \
# -o models/victim/pretrained-$DATASET-$MODEL_NAME \
# --pretrained_dir downloaded_models \
# -e 30 \
# --log-interval 25 \
# --lr $LR \
# --lr-step 20 \

# done



