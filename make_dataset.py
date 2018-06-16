import argparse
import numpy as np
import os
import pickle

import gym

def save_dataset(env_name, train, test):
    if not os.path.isdir("./data"):
        os.mkdir("./data/")

    with open("./data/{}_train.pickle".format(env_name), "wb") as f:
        print("train: ", train.shape)
        pickle.dump(train, f)
    with open("./data/{}_test.pickle".format(env_name), "wb") as f:
        print("test: ", test.shape)
        pickle.dump(test, f)


def collect_data(env_name):
    try:
        env = gym.make(env_name)
    except gym.error.Error:
        print("Wrong gym name")
        return

    n_episodes = 10
    max_step = 100
    obs_list = []
    for _ in range(n_episodes):
        step = 0
        _ = env.reset()
        while True:
            action = env.action_space.sample()
            obs, r, terminal, info = env.step(action)
            assert obs.shape == (210, 160, 3) or obs.shape == (250, 160, 3)  # gym observation space
            obs = obs.transpose(2, 0, 1)
            obs = obs.astype(np.float32)
            obs_list.append(obs/255.)

            step += 1
            if step >= max_step or terminal:
                break
    return obs_list


def make_dataset(env_name):
    print("env_name: ", env_name)
    obs_list = collect_data(env_name)
    assert len(obs_list) > 0

    np.random.shuffle(obs_list)
    train = []
    test = []
    for i, obs in enumerate(obs_list):
        if i%10 == 0:
            test.append(obs)
        else:
            train.append(obs)
    train = np.array(train)
    test = np.array(test)

    save_dataset(env_name, train, test)


def arg_parser():
    parser = argparse.ArgumentParser(description="chainer-RL-autoencoder")
    parser.add_argument("--env", "-e", type=str, help="gym env name(str)",
                        default="Bowling-v0")
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()
    make_dataset(args.env)

