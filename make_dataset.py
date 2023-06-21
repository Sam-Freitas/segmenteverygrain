import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tqdm import tqdm, trange
from glob import glob 
import segmenteverygrain as seg
import os
import matplotlib.pyplot as plt
from importlib import reload
from natsort import natsorted
import time
reload(seg)

from tensorflow.test import gpu_device_name

## create patches from images

def del_dir_contents(path_to_dir):
    time.sleep(0.1)
    files = glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)

print(gpu_device_name())

image_dir = r'Lightsaver_dataset\images'
mask_dir = r'Lightsaver_dataset\masks'
tiles_dir = r'Lightsaver_dataset\image_tiles'
mask_tiles_dir = r'Lightsaver_dataset\mask_tiles'
checkpoints_path = r'Unet_checkpoints/checkpoints/'
training_dump_path = r'training_dumps'
os.makedirs(checkpoints_path, exist_ok=True)

images = natsorted(glob(os.path.join(image_dir, "*.png")))
labels = natsorted(glob(os.path.join(mask_dir, "*.png")))

resizing = [4] # scale

img_size = 256
stride = 256
del_dir_contents(tiles_dir)
start_no = 0
sample_weight = []
for image in tqdm(images):
    # Load the large image
    large_image = tf.keras.preprocessing.image.load_img(image)
    # Convert the image to a tensor
    large_image = tf.keras.preprocessing.image.img_to_array(large_image)
    large_image = (large_image/np.max(large_image))*255
    # Reshape the tensor to have a batch size of 1
    large_image = tf.reshape(large_image, [1, *large_image.shape])
    # Extract patches from the large image
    for resize_num in resizing:
        resize_to = int(np.max(large_image.numpy().shape)/resize_num)
        temp_image = tf.image.resize(large_image,(resize_to,resize_to))

        patches = tf.image.extract_patches(
            images=temp_image,
            sizes=[1, img_size, img_size, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Reshape the patches tensor to have a batch size of -1
        patches = tf.reshape(patches, [-1, img_size, img_size, 3])
        # Write patches to files
        for i in range(patches.shape[0]):
            im = np.asarray(patches[i,:,:,:]).astype('uint8')
            imname = os.path.join(tiles_dir,'im%03d.png'%(start_no + i))
            im = Image.fromarray(im.astype(np.uint8))
            im.save(imname)
            sample_weight.append(resize_num)
        start_no = start_no + patches.shape[0]
np.savetxt(os.path.join(os.getcwd(),'Lightsaver_dataset','sample_weights.txt'),np.asarray(sample_weight)**2,fmt = '%10.5f')

del_dir_contents(mask_tiles_dir)
kernel = np.ones((5,5), np.uint8)
start_no = 0
for image in tqdm(labels):

    large_image = cv2.imread(image)[:,:,0]/255
    large_image = large_image + (large_image-cv2.erode(large_image,kernel))
    large_image = tf.keras.utils.to_categorical(large_image)
    large_image = tf.reshape(large_image, [1, *large_image.shape])

    for resize_num in resizing:
        resize_to = int(np.max(large_image.numpy().shape)/resize_num)
        temp_image = tf.image.resize(large_image,(resize_to,resize_to))

        # Extract patches from the large image
        patches = tf.image.extract_patches(
            images=temp_image,
            sizes=[1, img_size, img_size, 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # Reshape the patches tensor to have a batch size of -1
        patches = tf.reshape(patches, [-1, img_size, img_size, 3])
        # Write patches to files
        for i in range(patches.shape[0]):
            im = np.asarray(patches[i,:,:,:]).astype('uint8')
            imname = os.path.join(mask_tiles_dir,'im%03d.png'%(start_no + i))
            im = Image.fromarray(im.astype(np.uint8))
            im.save(imname)
        start_no = start_no + patches.shape[0]

print("EOF")