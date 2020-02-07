# Train orignal cifar10 from scratch
python run.py --checkpoint=../log/cifar10/ --epochs=300 --net=cifar10Net --wdecay=0.0005 --decaystep=100 --logdir=../log/cifar10/ --lr=0.1 --decay=0.1
# Finetune from pre-trained model
python run.py --admm --restore --checkpoint paramsADMM --restoredir params --epochs 100
# ADMM train from pre-trained model
# python run.py --net=cifar10Net --admm --restore --modelfile=../checkpoint/cifar10/cifar10VGG11orig.npz --epochs=300
 python run.py --net=cifar10Net --admm --restore --modelfile=../checkpoint/cifar10/cifar10VGG11orig.npz --epochs=300 --fftsize=8

# Finetune in frequency domain
# python run.py --restore --test
python run.py --restore --ffttrain --fftsize=8 --checkpoint=../checkpoint/cifar10/ --net=cifar10Net --lr=0.01 --wdecay=0.0 --modelfile=../checkpoint/cifar10/cifar10VGG11admm.npz --test

