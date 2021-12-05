import argparse
import glob
import os
import random

import numpy as np
import shutil

from utils import get_module_logger
from random import shuffle


def split(data_dir):
    """
    Create three splits from the processed records. The files should be moved to new folders in the 
    same directory. This folder should be named train, val and test.

    args:
        - data_dir [str]: data directory, /home/workspace/data/waymo
    """
    
    # TODO: Split the data present in `/home/workspace/data/waymo/training_and_validation` into train and val sets.
    # You should move the files rather than copy because of space limitations in the workspace.
    training_files = glob.glob(data_dir + '/training_and_validation/*.tfrecord')
    shuffle(training_files)
    num = len(training_files)
    
    # create the directry
    for _dir in ["train", "eval", "test"]:
        dir_path = "{}/{}".format(data_dir, _dir)
        dir_path = os.path.abspath(dir_path)
        os.makedirs(dir_path, exist_ok=True)
    
    #split out the training part, 0.7
    start = 0
    end = start + int(0.7 * num)
    for file_path in training_files[start:end]:
        dst_path = "{}/train/".format(data_dir)
        shutil.move(file_path, dst_path)
    #split out the eval part, 0.2
    start = end
    end = start + int(0.2 * num)
    for file_path in training_files[start:end]:
        dst_path = "{}/eval/".format(data_dir)
        shutil.move(file_path, dst_path)
    
    #split out the test part, 0.1
    for file_path in training_files[end:]:
        dst_path = "{}/test/".format(data_dir)
        shutil.move(file_path, dst_path)
    

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--data_dir', required=True,
                        help='data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.data_dir)