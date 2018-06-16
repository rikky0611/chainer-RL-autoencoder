# chainer-RL-autoencoder
Chainer Implementation of AutoEncoder for Deep Reinforcement Learning   
You can use this repository to make an encoder for visual Deep Reinforcement Learning.

## Requirements
gym == 0.9.2    
chainer >= 3.3.0    
numpy >= 1.14.0   
cupy >= 4.0.0 (if you use gpu)

## Prepare Dataset
You can just run, for example   

`python make_dataset.py --env Bowling-v0`

The default configuration is `10 episodes` and `100 max steps`.   
You can actually skip this command since it is implemented and called in `train.py`

## How to Run
You can just run, for example

`python train.py --env Bowling-v0`

The default network is a convolutional autoencoder. You can modify in `net.py`
### Arguments

- `--env`: gym environment name
- `--n_hidden`: hidden layer dimension which you later use as observation space for your RL agent
- `--batchsize`: batchsize for training
- `--epoch`: maximum epochs for training
- `--gpu`: your gpu number (negative if you use cpu)
- `--out`: out directory
- `--snapshot_interval`: interval iterations for snapshot
- `--display_interval`: interval iterations for display image and log

### Example result

<img width="933" alt="sample_result" src="https://user-images.githubusercontent.com/12772049/41498558-a2eb4656-71ab-11e8-8a49-e3f17016c603.png">
