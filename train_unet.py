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
reload(seg)

from tensorflow.test import gpu_device_name

## create patches from images

def del_dir_contents(path_to_dir):
    files = glob(os.path.join(path_to_dir,'*'))
    for f in files:
        os.remove(f)

def process_mask(mask,kernel = np.ones((5,5), np.uint8)):
    out + (mask-cv2.erode(mask,kernel))
    return out

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


# del_dir_contents(tiles_dir)
# start_no = 0
# for image in tqdm(images):
#     # Load the large image
#     large_image = tf.keras.preprocessing.image.load_img(image)
#     # Convert the image to a tensor
#     large_image = tf.keras.preprocessing.image.img_to_array(large_image)
#     large_image = (large_image/np.max(large_image))*255
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
    
# del_dir_contents(mask_tiles_dir)
# kernel = np.ones((5,5), np.uint8)
# start_no = 0
# for image in tqdm(labels):
#     # Load the large image
#     # large_image = tf.keras.preprocessing.image.load_img(image)
#     # # Convert the image to a tensor
#     # large_image = tf.keras.preprocessing.image.img_to_array(large_image)
#     # large_image = large_image[:,:,0,np.newaxis] # only keep one layer and add a new axis

#     large_image = cv2.imread(image)[:,:,0]/255
#     large_image = large_image + (large_image-cv2.erode(large_image,kernel))
#     large_image = tf.keras.utils.to_categorical(large_image)
#     large_image = large_image[np.newaxis,:]
    
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
#         imname = os.path.join(mask_tiles_dir,'im%03d.png'%(start_no + i))
#         im = Image.fromarray(im.astype(np.uint8))
#         im.save(imname)
#     start_no = start_no + patches.shape[0]

image_files = natsorted(glob(os.path.join(tiles_dir, "*.png")))
mask_files = natsorted(glob(os.path.join(mask_tiles_dir, "*.png")))

batch_size = 128
shuffle_buffer_size = 1000   

train_idx = np.arange(len(image_files))[0:int(len(image_files)/2)] # first 50%
val_idx = np.arange(len(image_files))[int(len(image_files)/2):(int(len(image_files)/2)+int(len(image_files)/4))] # next 25%
test_idx = np.arange(len(image_files))[(int(len(image_files)/2)+int(len(image_files)/4)):] # remaining 25%

# # split half into training
# train_idx = np.random.choice(np.arange(len(image_files)), size = int(len(image_files)/2) , replace=False)
# # get the rest
# idx = np.setdiff1d(np.arange(len(image_files)), train_idx)
# # get val files (25% of total)
# val_idx = np.random.choice(idx, size = int(len(image_files)/4) , replace=False)
# # get test files (25% of total)
# test_idx = np.setdiff1d(val_idx, train_idx)

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

model = seg.Unet()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer=optimizer, loss=seg.weighted_crossentropy, metrics=["accuracy"])
# model.summary()

epochs = 1000

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,save_weights_only=True,verbose=1,save_best_only = True)
es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=epochs, restore_best_weights=True)

class TestAtEpochEnd(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        # Initialize the best as infinity.
        self.best = np.Inf
    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            field_names = list(logs.keys())
            try:
                os.remove('training_results.txt')
            except:
                pass
            with open('training_results.txt', 'a') as fd:
                fd.write(str(field_names) + '\n')
                fd.write(str(np.round(np.asarray(list(logs.values())),6)) + '\n')
        else:
            with open('training_results.txt', 'a') as fd:
                fd.write(str(np.round(np.asarray(list(logs.values())),6)) + '\n')
        
        current = logs.get("val_loss")
        os.makedirs(training_dump_path, exist_ok=True)
        if current < self.best:
            eval_results = model.evaluate(test_dataset)
            print('EVAL LOSS:', round(eval_results[0],4), 'EVAL ACC:', round(eval_results[1],4))
            self.best = current
            del_dir_contents(training_dump_path)
            counter = 0
            for i,element in enumerate(test_dataset.as_numpy_iterator()):
                test_x,test_y = element
                prediction = self.model.predict(element[0],verbose = 0)
                for j in range(prediction.shape[0]):
                    temp = (cv2.hconcat([test_x[j]/np.max(test_x[j]),test_y[j][:,:,::-1],prediction[j][:,:,::-1]])*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(training_dump_path,str(counter) + '.jpg'),temp)
                    counter = counter + 1
                if i > 2:
                    break
        else:
            print('Loss did not improve')

# model.load_weights(r'Unet_checkpoints/checkpoints/')
history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset,callbacks=[cp_callback,es_callback,TestAtEpochEnd()])

model.evaluate(test_dataset)
model.save_weights('Unet_checkpoints/weights/')

# testing
model2 = seg.Unet()
model2.load_weights(r'Unet_checkpoints/weights/')

counter = 0
for i,element in enumerate(test_dataset.as_numpy_iterator()):
    test_x,test_y = element
    prediction = model2.predict(element[0],verbose = 0)
    for j in range(prediction.shape[0]):
        temp = (cv2.hconcat([test_x[j]/np.max(test_x[j]),test_y[j][:,:,::-1],prediction[j][:,:,::-1]])*255).astype(np.uint8)
        cv2.imwrite(os.path.join(training_dump_path,str(counter) + '.jpg'),temp)
        counter = counter + 1

# image = cv2.imread(val_images[best_test_idx])
# mask = cv2.imread(val_masks[best_test_idx])
# # image = tf.image.decode_png(image, channels=3)

# out = model2.predict(image[np.newaxis,:])
# out = out.squeeze()

# a = np.concatenate([image/255,mask/np.max(mask),out],axis = 1)
# plt.imshow(a)
# plt.show()

print('EOF')