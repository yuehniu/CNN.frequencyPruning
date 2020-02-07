CUDA_VISIBLE_DEVICES=2 python run.py \
--epochs=2400 \
--decaystep=300 \
--checkpoint=checkpoint/cifar10 \
--restoredir=checkpoint/cifar10 \
--logdir=log/cifar10 \
--net=cifar10Net \
--lr=0.0005 \
--admm \
--rho=0.005 \
--restore
#CUDA_VISIBLE_DEVICES=0 python run.py \
#--epochs=30 \
#--checkpoint=checkpoint/cifar10 \
#--restoredir=checkpoint/cifar10 \
#--logdir=log/cifar10 \
#--net=cifar10Net \
#--lr=0.0001 \
#--wdecay=0.00001 \
#--ffttrain \
#--restore
