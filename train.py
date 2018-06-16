import argparse
import numpy as np
import os
import pickle

import chainer
from chainer import training
from chainer.training import extensions
import chainer.links as L
import chainer.functions as F

from net import AutoEncoder
from visualizer import out_generated_image
from make_dataset import make_dataset

def argparser():
    parser = argparse.ArgumentParser(description='AutoEncoder for atari image')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=1000,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--snapshot_interval', type=int, default=10000,
                        help='Interval of snapshot')
    parser.add_argument('--display_interval', type=int, default=500,
                        help='Interval of displaying log to console')
    parser.add_argument('--env', type=str, default="Bowling-v0",
                        help="atari env")
    parser.add_argument('--n_hidden', type=int, default=100,
                        help="hidden layer dimension")
    args = parser.parse_args()
    print('Env: {}'.format(args.env))
    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# Epoch: {}'.format(args.epoch))
    print('# Snapshot_interval: {}'.format(args.snapshot_interval))
    print('# Display_interval: {}'.format(args.display_interval))
    print('# N_hidden: {}'.format(args.n_hidden))
    print('')
    return args


def main():
    args = argparser()

    # Check data
    train_path = "./data/{}_train.pickle".format(args.env)
    test_path = "./data/{}_test.pickle".format(args.env)
    if os.path.exists(train_path) and os.path.exists(test_path):
        print("load data from ./data/")
    else:
        print("make dataset ...")
        make_dataset(env_name=args.env)

    # Setup iterator
    train = pickle.load(open(train_path, "rb"))
    train = [(t, t) for t in train]
    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test = pickle.load(open(test_path, "rb"))
    test = [(t, t) for t in test]
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize, repeat=False, shuffle=False)

    if test[0][0].shape == (3, 210, 160):
        n_linear_dim = 30704
    elif test[0][0].shape == (3, 250, 160):
        n_linear_dim = 36784
    else:
        raise Exception

    # Set up a neural network to train
    ae = AutoEncoder(n_hidden=args.n_hidden, n_linear_dim=n_linear_dim)
    if args.gpu >= 0:
        chainer.cuda.check_cuda_available()
        chainer.cuda.get_device_from_id(args.gpu).use()
        xp = chainer.cuda.cupy
        ae.to_gpu()
    else:
        xp = np
    ae = L.Classifier(ae, lossfun=F.mean_squared_error)
    ae.compute_accuracy = False

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
    optimizer.setup(ae)

    # Setup trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    # Extenstion
    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    trainer.extend(extensions.snapshot(
                filename='snapshot_{0}_iter-{1}'.format(args.env, updater.iteration)),
                trigger=snapshot_interval)
    trainer.extend(extensions.Evaluator(test_iter, ae, device=args.gpu), name='val')
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'main/loss', 'val/main/loss'
    ]), trigger=display_interval)
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(out_generated_image(ae, updater, test[0][0], args.env), trigger=display_interval)

    # Run trainer
    trainer.run()

if __name__ == '__main__':
    main()
