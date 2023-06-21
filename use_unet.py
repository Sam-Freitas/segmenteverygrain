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
batch_size = 128
shuffle_buffer_size = 1000   

image_dir = r"C:\Users\LabPC2\Documents\GitHub\segmenteverygrain\Lightsaver_dataset\images"
image_dir = r"Y:\Users\Raul Castro\Microscopes\Leica Fluorescence Stereoscope\GFP Marker New set 2\2021-04-08\Exported"

images = natsorted(glob(os.path.join(image_dir, "*.tif")) + glob(os.path.join(image_dir, "*.png")))

tif_checker = np.sum(np.asarray([1 if img[-3:]=='tif' else 0 for img in images]))
if tif_checker > 0:
    images = seg.dump_to_pngs(images,preprocess_fluor_img=True, remove_scale_bar=True)

test_dataset = tf.data.Dataset.from_tensor_slices((images))
test_dataset = test_dataset.map(seg.load_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# test_dataset = test_dataset.shuffle(shuffle_buffer_size)

resizing = [1,2] # scale

img_size = 512
patch_size = 512
stride_size = 128

# testing
model = seg.Unet(input_shape=(img_size,img_size,3))
model.load_weights(r'Unet_checkpoints/checkpoints/').expect_partial()

os.makedirs(os.path.join(os.getcwd(),'testing_dump'), exist_ok=True)
del_dir_contents(os.path.join(os.getcwd(),'testing_dump'))
counter = 0
for i,element in enumerate(test_dataset.as_numpy_iterator()):
    for each_subelement in tqdm(element):
        predictions = []
        plt.close('all')
        export_image = each_subelement-np.min(each_subelement)
        export_image = export_image/np.max(export_image)
        for resize_num in resizing:
            resize_to = int(np.max(each_subelement.shape)/resize_num)
            temp_image = tf.image.resize(each_subelement,(resize_to,resize_to))
            input = temp_image
                    # Extract patches from the large image
            patches = tf.image.extract_patches(
                images=input[np.newaxis,:],
                sizes=[1, patch_size, patch_size, 1],
                strides=[1, stride_size, stride_size, 1],
                rates=[1, 1, 1, 1],
                padding='VALID'
            )

            patches = tf.reshape(patches, [-1, patch_size, patch_size, 3])
            prediction_on_patches = model.predict(patches, verbose = 0, batch_size = 16)

            this_prediction = recon_im(prediction_on_patches, input.shape[0], input.shape[1], input.shape[2], stride_size)
            predictions.append(cv2.resize(this_prediction, dsize=(np.max(each_subelement.shape), np.max(each_subelement.shape)), interpolation=cv2.INTER_CUBIC)/resize_num)
        prediction_averaged = np.mean(np.asarray(predictions),axis = 0)
        post_processed_prediction = tf.keras.utils.to_categorical(np.argmax(prediction_averaged,axis = -1),num_classes=3)
        cv2.imwrite(os.path.join(os.getcwd(),'testing_dump',str(counter) + '.jpg'),np.concatenate((export_image,post_processed_prediction),axis = 1)[:,:,::-1]*255)
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