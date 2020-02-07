EPOCH=(150 100 100 100 100)
LR=(0.1 0.05 0.01 0.005 0.001)
WD=(0.0005 0.0003 0.0001 0.00005 0.00001)
RS=(false true true true true)

for((i=0; i<${#EPOCH[@]}; i++)); do
  epoch=${EPOCH[i]}
  lr=${LR[i]}
  wd=${WD[i]}
  rs=${RS[i]}

  if [ "$rs" = false ]; then
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --epochs=${epoch} \
    --decaystep=100 \
    --checkpoint=checkpoint/cifar10 \
    --restoredir=checkpoint/cifar10 \
    --logdir=log/cifar10 \
    --net=cifar10Net \
    --lr=${lr} \
    --wdecay=${wd}
  else
    CUDA_VISIBLE_DEVICES=0 python run.py \
    --epochs=${epoch} \
    --decaystep=100 \
    --checkpoint=checkpoint/cifar10 \
    --restoredir=checkpoint/cifar10 \
    --logdir=log/cifar10 \
    --net=cifar10Net \
    --lr=${lr} \
    --wdecay=${wd} \
    --restore
  fi
done
