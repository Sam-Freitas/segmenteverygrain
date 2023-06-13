import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import segmenteverygrain as seg

size = 1024
k_size = 256
axes_1_2_size = int(np.sqrt((size * size) / (k_size * k_size)))

# Define a placeholder for image (or load it directly if you prefer) 
img = cv2.imread(r"C:\Users\LabPC2\Documents\GitHub\segmenteverygrain\Lightsaver_dataset\images\3.png")[np.newaxis,:]/255

# Extract patches
patches = tf.image.extract_patches(
    images=img,
    sizes=[1, k_size,k_size, 1],
    strides=[1, k_size,k_size, 1],
    rates=[1, 1, 1, 1],
    padding='SAME'
)
patches = tf.reshape(patches, [-1, k_size,k_size, 3])

model = seg.Unet()
model.load_weights(r'Unet_checkpoints/weights/').expect_partial()
prediction = model.predict(patches)
# Reconstruct the image back from the patches
# First separate out the channel dimension
reconstruct = tf.reshape(prediction, (1, axes_1_2_size, axes_1_2_size, k_size, k_size, 3)) 
# Tranpose the axes (I got this axes tuple for transpose via experimentation)
reconstruct = tf.transpose(reconstruct, (0, 1, 3, 2, 4, 5))
# Reshape back
reconstruct = tf.reshape(reconstruct, (size, size, 3))

# Plot the reconstructed image to verify
plt.imshow(reconstruct)
plt.show()