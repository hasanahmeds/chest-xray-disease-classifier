import json
import shutil
import os
import pickle
from callback import MultipleClassAUROC, MultiGPUModelCheckpoint
from configparser import ConfigParser
from generator import AugmentedImageSequence
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.utils import multi_gpu_model
from models.keras import ModelFactory
from utility import get_sample_counts
from weights import get_class_weights
from augmenter import augmenter


def main():
    # parser config
    config_file = "./config.ini"
    cp = ConfigParser()
    cp.read(config_file)

    # default config
    output_dir = cp["DEFAULT"].get("output_dir")
    image_source_dir = cp["DEFAULT"].get("image_source_dir")
    base_model_name = cp["DEFAULT"].get("base_model_name")
    class_names = cp["DEFAULT"].get("class_names").split(",")

    # train config
    use_base_model_weights = cp["TRAIN"].getboolean("use_base_model_weights")
    use_trained_model_weights = cp["TRAIN"].getboolean("use_trained_model_weights")
    use_best_weights = cp["TRAIN"].getboolean("use_best_weights")
    output_weights_name = cp["TRAIN"].get("output_weights_name")
    epochs = cp["TRAIN"].getint("epochs")
    batch_size = cp["TRAIN"].getint("batch_size")
    initial_learning_rate = cp["TRAIN"].getfloat("initial_learning_rate")
    generator_workers = cp["TRAIN"].getint("generator_workers")
    image_dimension = cp["TRAIN"].getint("image_dimension")
    train_steps = cp["TRAIN"].get("train_steps")
    patience_reduce_lr = cp["TRAIN"].getint("patience_reduce_lr")
