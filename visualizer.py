import os
import sys

import numpy as np
import pickle
from PIL import Image

import chainer
import chainer.cuda


def out_generated_image(ae, updater, img, env_name):

    @chainer.training.make_extension()
    def make_image(trainer):
        xp = ae.xp
        model = ae.predictor
        ch = img.shape[0]
        W = img.shape[1]
        H = img.shape[2]
        x = xp.array(img.reshape(1, ch, W, H))
        with chainer.using_config("Train", False):
            h = model.encode(x)
            y = model.decode(h)
        # convert to image
        y = chainer.cuda.to_cpu(y.data)
        x = chainer.cuda.to_cpu(x)
        x = np.clip(x * 255, 0.0, 255.0).astype(np.uint8) 
        y = np.clip(y * 255, 0.0, 255.0).astype(np.uint8)
        # convert shape from (1, 3, 210, 160) to (1, 210, 160, 3)
        x = np.array(x.reshape(ch, W, H).transpose(1, 2, 0), dtype=np.uint8)
        y = np.array(y.reshape(ch, W, H).transpose(1, 2, 0), dtype=np.uint8)
        original_path = "./image/{}_org.png".format(env_name)
        generated_path = "./image/{}_{}.png".format(env_name, updater.iteration)
        Image.fromarray(x).save(original_path)
        Image.fromarray(y).save(generated_path)

    return make_image
