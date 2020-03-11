"""
This file is use to process command line arguments when commander is called
"""

import sys
import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CSRNet')

    # Experiment settings
    # name of the experiment
    parser.add_argument('--name', default='CSRNet_cell', type=str,
                        help='name of experiment')
    parser.add_argument('--test', dest='test', action='store_true', default=False,
                        help='To only run inference on test set')

    # Dataset
    # main folder for data storage
    parser.add_argument('--root-dir', type=str, default=None)
    parser.add_argument('--train_csv', type=str, default=None)
    parser.add_argument('--test_csv', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='density',
                        help='dataset to use [dot, density, count]')

    # About the model
    parser.add_argument('--arch', type=str,
                        help='type of architecture to be used, e.g. CSRNet', default='CSRNet')
    parser.add_argument('--model-backend', type=str, default='B',
                        help='type of model to be used. Particular instance of a given architecture, e.g. A or dilation as \'1, 1, 1, 2\'')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='which checkpoint to resume from. possible values["latest", "best", epoch]')
    parser.add_argument('--pretrained', action='store_true',
                        default=True, help='use pre-trained model from VGG')

    # Loader
    # number of workers for the dataloader
    parser.add_argument('-j', '--workers', type=int, default=4)

    # Training
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--step', type=int, default=20,
                        help='frequency of updating learning rate, given in epochs')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=400, metavar='N',
                        help='number of epochs to train (default: 400)')
    parser.add_argument('--optimizer', default='sgd', type=str,
                        help='name of the optimizer')
    parser.add_argument('--scheduler', default='StepLR', type=str,
                        help='name of the learning rate scheduler')
    parser.add_argument('--lr', type=float, default=1e-7, metavar='LR',
                        help='learning rate (default: 1e-7)')
    parser.add_argument('--momentum', type=float, default=0.95, metavar='M',
                        help='sgd momentum (default: 0.95)')
    parser.add_argument('--weight-decay', '--wd', default=5*1e-4, type=float,
                        metavar='W', help='weight decay (default: 5*1e-4)')
    parser.add_argument('--lr-decay', default=0.995, type=float,
                        metavar='lrd', help='learning rate decay (default: 0.995)')
    parser.add_argument('--criterion', default='mse', type=str,
                        help='criterion to optimize, default is MSE')

    # Other
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--short-run', action='store_true',
                        default=False, help='running only over few mini-batches for debugging purposes')
    parser.add_argument('--disable-cuda', action='store_true', default=False,
                        help='disables CUDA training / using only CPU')
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', default=False,
                        help='Use tensorboard to track and plot')

    # From CSRNet
    # parser.add_argument('train_json', metavar='TRAIN',
    #                     help='path to train json')
    # parser.add_argument('test_json', metavar='TEST',
    #                     help='path to test json')

    # parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
    #                     help='path to the pretrained model')

    # parser.add_argument('gpu', metavar='GPU', type=str,
    #                     help='GPU id to use.')

    # parser.add_argument('task', metavar='TASK', type=str,
    #                     help='task id to use.')

    args = parser.parse_args()

    # update args
    # args.data_dir = '{}/{}'.format(args.root_dir, args.dataset)
    args.log_dir = '{}/runs/{}/'.format(args.data_dir, args.name)
    args.res_dir = '%s/runs/%s/res' % (args.data_dir, args.name)
    args.out_pred_dir = '%s/runs/%s/pred' % (args.data_dir, args.name)

    args.cuda = not args.disable_cuda and torch.cuda.is_available()
    args.device = 'cuda' if args.cuda else 'cpu'

    assert args.data_dir is not None

    print(' '.join(sys.argv))
    print(args)

    return args
