import os
import glob
import sys
import numpy as np
import pickle
import tensorflow as tf


def generate_synthetic_face():
    """ load feature directions """
    path_feature_direction = './asset_results/pg_gan_celeba_feature_direction_40'

    pathfile_feature_direction = glob.glob(os.path.join(path_feature_direction, 'feature_direction_*.pkl'))[-1]

    with open(pathfile_feature_direction, 'rb') as f:
        feature_direction_name = pickle.load(f)

    feature_direction = feature_direction_name['direction']
    feature_name = feature_direction_name['name']
    num_feature = feature_direction.shape[1]

    """ load gan model """

    # path to model code and weight
    path_pg_gan_code = './src/model/pggan'
    path_model = './asset_model/karras2018iclr-celebahq-1024x1024.pkl'
    sys.path.append(path_pg_gan_code)

    """ create tf session """
    yn_CPU_only = False

    if yn_CPU_only:
        config = tf.ConfigProto(device_count = {'GPU': 0}, allow_soft_placement=True)
    else:
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True

    sess = tf.InteractiveSession(config=config)

    try:
        with open(path_model, 'rb') as file:
            G, D, Gs = pickle.load(file)
    except FileNotFoundError:
        print('before running the code, download pre-trained model to project_root/asset_model/')
        raise

    num_latent = Gs.input_shapes[0][1]

    latents = np.random.randn(1, *Gs.input_shapes[0][1:])
    # Generate dummy labels
    dummies = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

    images = Gs.run(latents, dummies)
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8)  # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1)  # NCHW => NHWC

    ##
    """ plot figure """
    #plt.imshow(images[0])
    #plt.show()

    return images[0]