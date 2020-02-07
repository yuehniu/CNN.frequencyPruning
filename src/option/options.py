"""Hyper parameter during train DNN"""
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--net', type=str, default='LeNet', help='Net model')
parser.add_argument('--batchsize', type=int, default=128, help='batch size during train and test')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs of training')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
parser.add_argument('--wdecay', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay', type=float, default=0.8, help='decay rate of learning rate')
parser.add_argument('--decaystep', type=int, default=20, help='decay step of learning rate')
parser.add_argument('--printstep', type=int, default=100, help='print train message after define steps')
parser.add_argument('--test', action="store_true", help='test pre-trained model')
parser.add_argument('--admm', action="store_true", help='train with admm')
parser.add_argument('--comratio', type=int, default=4, help='compression ratio')
parser.add_argument('--rho', type=float, default=0.01, help='ADMM learning factor')
parser.add_argument('--restore', action="store_true", help='restore params from pre-trained model')
parser.add_argument('--modelfile', type=str, default='../checkpoint/cifar10VGG11admm.npz', help='restore param directory')
parser.add_argument('--checkpoint', type=str, default='../checkpoint/cifar10/', help='checkpoint file path')
parser.add_argument('--logdir', type=str, default='../log/cifar10/', help='run log file during train and test')
parser.add_argument('--fftsize', type=int, default=16, help='FFT size when doing frequency-domain conv.')
parser.add_argument('--ffttrain', action="store_true", help='Finetune model in frequency domain.')
parser.add_argument('--augdev', type=float, default=0.1, help='std dev of noise in data augmentation.')

opt = parser.parse_args()
