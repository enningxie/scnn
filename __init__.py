import numpy as np
import tensorflow as tf


from data_reader import image_data_set

train_images_path = '/home/enningxie/Documents/Codes/python/scnn/dataset/train_img'
train_gt_path = '/home/enningxie/Documents/Codes/python/scnn/dataset/train_gt'
train_data = image_data_set(train_images_path, train_gt_path,
                                do_shuffle=True)