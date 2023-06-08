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

## create patches from images

print(gpu_device_name())

image_dir = r'Lightsaver_dataset\images'
mask_dir = r'Lightsaver_dataset\masks'
tiles_dir = r'Lightsaver_dataset\image_tiles'
mask_tiles_dir = r'Lightsaver_dataset\mask_tiles'

images = natsorted(glob(os.path.join(image_dir, "*.png")))
labels = natsorted(glob(os.path.join(mask_dir, "*.png")))

# start_no = 0
# for image in tqdm(images):
#     # Load the large image
#     large_image = tf.keras.preprocessing.image.load_img(image)
#     # Convert the image to a tensor
#     large_image = tf.keras.preprocessing.image.img_to_array(large_image)
#     # Reshape the tensor to have a batch size of 1
#     large_image = tf.reshape(large_image, [1, *large_image.shape])
#     # Extract patches from the large image
#     patches = tf.image.extract_patches(
#         images=large_image,
#         sizes=[1, 256, 256, 1],
#         strides=[1, 128, 128, 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID'
#     )
#     # Reshape the patches tensor to have a batch size of -1
#     patches = tf.reshape(patches, [-1, 256, 256, 3])
#     # Write patches to files
#     for i in range(patches.shape[0]):
#         im = np.asarray(patches[i,:,:,:]).astype('uint8')
#         imname = os.path.join(tiles_dir,'im%03d.png'%(start_no + i))
#         im = Image.fromarray(im.astype(np.uint8))
#         im.save(imname)
#     start_no = start_no + patches.shape[0]
    
# start_no = 0
# for image in tqdm(labels):
#     # Load the large image
#     large_image = tf.keras.preprocessing.image.load_img(image)
#     # Convert the image to a tensor
#     large_image = tf.keras.preprocessing.image.img_to_array(large_image)
#     large_image = large_image[:,:,0,np.newaxis] # only keep one layer and add a new axis
#     # Reshape the tensor to have a batch size of 1
#     large_image = tf.reshape(large_image, [1, *large_image.shape])
#     # Extract patches from the large image
#     patches = tf.image.extract_patches(
#         images=large_image,
#         sizes=[1, 256, 256, 1],
#         strides=[1, 128, 128, 1],
#         rates=[1, 1, 1, 1],
#         padding='VALID'
#     )
#     # Reshape the patches tensor to have a batch size of -1
#     patches = tf.reshape(patches, [-1, 256, 256, 1])
#     # Write patches to files
#     for i in range(patches.shape[0]):
#         im = np.asarray(patches[i,:,:,0]).astype('uint8')
#         imname = os.path.join(mask_tiles_dir,'im%03d.png'%(start_no + i))
#         im = Image.fromarray(im.astype(np.uint8))
#         im.save(imname)
#     start_no = start_no + patches.shape[0]

image_files = natsorted(glob(os.path.join(tiles_dir, "*.png")))
mask_files = natsorted(glob(os.path.join(mask_tiles_dir, "*.png")))

batch_size = 128
shuffle_buffer_size = 1000

# split half into training
train_idx = np.random.choice(np.arange(len(image_files)), size = int(len(image_files)/2) , replace=False)
# get the rest
idx = np.setdiff1d(np.arange(len(image_files)), train_idx)
# get val files (25% of total)
val_idx = np.random.choice(idx, size = int(len(image_files)/4) , replace=False)
# get test files (25% of total)
test_idx = np.setdiff1d(val_idx, train_idx)

# create arrays of training, validation, and test files (these are filenames)
train_images = np.array(image_files)[train_idx]
val_images = np.array(image_files)[val_idx]
test_images = np.array(image_files)[test_idx]

train_masks = np.array(mask_files)[train_idx]
val_masks = np.array(mask_files)[val_idx]
test_masks = np.array(mask_files)[test_idx]

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
train_dataset = train_dataset.map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_masks))
val_dataset = val_dataset.map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_masks))
test_dataset = test_dataset.map(seg.load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(shuffle_buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# example_img = train_images[32]
# example_mask = train_masks[32]
# img = cv2.imread(example_img)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # need to convert from BGR to RGB
# mask = cv2.imread(example_mask)
# seg.plot_images_and_labels(img, mask)

model = seg.Unet()
model.compile(optimizer=Adam(), loss=seg.weighted_crossentropy, metrics=["accuracy"])
model.summary()

history = model.fit(train_dataset, epochs=200, validation_data=val_dataset)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xlim([50,200])
plt.ylim([0.7, 0.72])

plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xlim([100,200])
plt.ylim([0.95, 0.982])

model.evaluate(test_dataset)
model.save_weights('seg_model')

print('EOF')