


for dataset in CIFAR10 CIFAR100 STL10 UTKFaceRace 
do
for arch in alexnet resnet18 vgg16_bn
do


# for dataset in CIFAR10 
# do
# for arch in resnet18 
# do

python -m nettailor_224.eval_sgx_nettailor_freivalds \
--arch $arch \
--dataset $dataset \
> nettailor_224/layer_analysis/${dataset}_${arch}.txt


done
done