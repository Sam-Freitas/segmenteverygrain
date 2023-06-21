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

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

## create patches from images

def del_dir_contents(path_to_dir):
    time.sleep(1)
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

resizing = [2] # scale

img_size = 256
stride = 256
 
image_files = natsorted(glob(os.path.join(tiles_dir, "*.png")))
mask_files = natsorted(glob(os.path.join(mask_tiles_dir, "*.png")))

batch_size = 256
shuffle_buffer_size = 1000   

train_idx = np.arange(len(image_files))[0:int(len(image_files)*0.5)] # first 50%
val_idx = np.arange(len(image_files))[int(len(image_files)*0.5):int(len(image_files)*0.75)] # next 25%
test_idx = np.arange(len(image_files))[int(len(image_files)*0.75):] # remaining 25%

temp_idx1 = test_idx[::2]
temp_idx2 = test_idx[1::2]
train_idx = np.concatenate((train_idx,temp_idx2),axis = 0)
test_idx = temp_idx1

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
test_dataset = test_dataset.map(seg.load_image_and_mask_nopreprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = seg.Unet(input_shape=(img_size,img_size,3))
# model = seg.attention_unet((img_size,img_size,3))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
model.compile(optimizer=optimizer, loss=seg.weighted_crossentropy, metrics=["accuracy",tf.keras.metrics.IoU(num_classes=3, target_class_ids=[0,1,2])])
# model.summary()

epochs = 10000

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoints_path,save_weights_only=True,verbose=1,save_best_only = True)
es_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=1000, restore_best_weights=True)

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
                fd.write(str(field_names))
                fd.write('\n' + str(np.round(np.asarray(list(logs.values())),6)) )
        else:
            with open('training_results.txt', 'a') as fd:
                fd.write('\n' + str(np.round(np.asarray(list(logs.values())),6)) )
        
        current = logs.get("val_loss")
        os.makedirs(training_dump_path, exist_ok=True)
        if current < self.best:
            with open('training_results.txt', 'a') as fd:
                fd.write('*')
            eval_results = model.evaluate(test_dataset)
            print('EVAL LOSS:', round(eval_results[0],4), 'EVAL ACC:', round(eval_results[1],4))
            self.best = current
            del_dir_contents(training_dump_path)
            counter = 0
            for i,element in enumerate(test_dataset.as_numpy_iterator()):
                test_x,test_y = element
                prediction = self.model.predict(element[0],verbose = 0)
                for j in range(prediction.shape[0]):
                    img_temp = (test_x[j]-np.min(test_x[j]))/(np.max(test_x[j]-np.min(test_x[j])) + 1)
                    temp = (cv2.hconcat([img_temp,test_y[j][:,:,::-1],prediction[j][:,:,::-1]])*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(training_dump_path,str(counter) + '.jpg'),temp)
                    counter = counter + 1
                if i > 4:
                    break
        else:
            print('Loss did not improve')

# model.load_weights(r'Unet_checkpoints/checkpoints/') 
history = model.fit(train_dataset, 
                    # sample_weight = sample_weight,
                    epochs=epochs, 
                    validation_data=val_dataset,
                    callbacks=[cp_callback,es_callback,TestAtEpochEnd()])

model.evaluate(test_dataset)
model.save_weights('Unet_checkpoints\weights')

# testing
model2 = seg.Unet(input_shape=(img_size,img_size,3))
# model2 = seg.attention_unet((img_size,img_size,3))
model2.load_weights(r'Unet_checkpoints\weights')

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