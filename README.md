# CNN.frequencyPruning
A frequency-domain pruning implementation using ADMM

## Environment

* Python: 3.6
* Tensorflow: 1.12 or higher (tf.dataset requires >= 1.14)
* CUDA: 8.0 or higher (tf1.14 requires CUDA >= 9)

## Dir
```
./
|--src
|    |--/models: model definitions.
|    |
|    |--/option: run options.
|    |
|    |--/scripts: run scripts.
|    |
|    |--/train: model train and test.
|    |
|    |--/utils: various utils, like layer defition.
|    
|--/log: train logs (not included in git).
|    
|--/result: some visualization of trained kernels.
|
|--/checkpoint: train checkpoints.
|
|--run.py: main function.
|
|--run.sh: run scripts
|
|--viewkernel.py: visualize kernels
```


## runing (CIFAR-10) 
For CIFAR-10, VGG16 was used to test compression performance ADMM based method.

### Train original CIFAR-10 model from scratch

```bash
python run.py --checkpoint=../log/cifar10/ --epochs=300 --net=cifar10Net --wdecay=0.0005 --decaystep=100 --logdir=../log/cifar10/ --lr=0.1 --decay=0.1
```

### ADMM finetune from well-trained spatial model

```bash
python run.py --admm --restore --checkpoint paramsADMM --modelfile=../checkpoint/cifar10/cifar10VGG11.npz --epochs 100 --net=cifar10Net
```

### ADMM test
```bash
python run.py --admm --restore --test --modelfile=../checkpoint/cifar10/cifar10VGG11.npz --net=cifar10Net
```

### Frequency-domain finetune from ADMM trained model
Remove near-zero values in convolutional kernels.
```bash
python run.py --restore --ffttrain --fftsize=8 --checkpoint=../checkpoint/cifar10/ --net=cifar10Net --lr=0.01 --wdecay=0.0 --modelfile=../checkpoint/cifar10/cifar10VGG11admm.npz --test
```