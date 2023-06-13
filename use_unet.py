import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from tqdm import tqdm, trange
from glob import glob 
import segmenteverygrain as seg
import os
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from importlib import reload
from natsort import natsorted
reload(seg)

from tensorflow.test import gpu_device_name
from patch_reconstructor.recon_from_patches import recon_im

## create patches from images

def del_dir_contents(path_to_dir):
    files = glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)


print(gpu_device_name())

image_dir = r"C:\Users\LabPC2\Documents\GitHub\segmenteverygrain\Lightsaver_dataset\images"

images = natsorted(glob(os.path.join(image_dir, "*.tif")) + glob(os.path.join(image_dir, "*.png")))

tif_checker = 0
for img in images:
    if img[-3:] == 'tif':
        tif_checker = tif_checker + 1

if tif_checker > 0:
    os.makedirs(os.path.join(os.getcwd(),'tif_dump'), exist_ok=True)
    del_dir_contents(os.path.join(os.getcwd(),'tif_dump'))
    for img in images:
        temp = cv2.imread(img)
        cv2.imwrite(os.path.join(os.getcwd(),'tif_dump',os.path.split(img)[-1][:-3] + 'png'),temp)

batch_size = 128
shuffle_buffer_size = 1000   

test_dataset = tf.data.Dataset.from_tensor_slices((images))
test_dataset = test_dataset.map(seg.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.shuffle(shuffle_buffer_size)

patch_size = 256
stride_size = 128

# testing
model = seg.Unet()
model.load_weights(r'Unet_checkpoints/weights/').expect_partial()

os.makedirs(os.path.join(os.getcwd(),'testing_dump'), exist_ok=True)
del_dir_contents(os.path.join(os.getcwd(),'testing_dump'))
counter = 0
for i,element in enumerate(test_dataset.as_numpy_iterator()):
    for each_subelement in element:
        input = each_subelement/np.max(each_subelement)
                # Extract patches from the large image
        patches = tf.image.extract_patches(
            images=input[np.newaxis,:],
            sizes=[1, patch_size, patch_size, 1],
            strides=[1, stride_size, stride_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        patches = tf.reshape(patches, [-1, patch_size, patch_size, 3])
        prediction_on_patches = model.predict(patches,verbose = 0)

        prediction = recon_im(prediction_on_patches, input.shape[0], input.shape[1], input.shape[2], stride_size)

        cv2.imwrite(os.path.join(os.getcwd(),'testing_dump',str(counter) + '.jpg'),np.concatenate((input,prediction),axis = 1)[:,:,::-1]*255)
        counter = counter + 1

        # if counter > 100:
        #     break




# image = cv2.imread(val_images[best_test_idx])
# mask = cv2.imread(val_masks[best_test_idx])
# # image = tf.image.decode_png(image, channels=3)

# out = model2.predict(image[np.newaxis,:])
# out = out.squeeze()

# a = np.concatenate([image/255,mask/np.max(mask),out],axis = 1)
# plt.imshow(a)
# plt.show()

print('EOF')