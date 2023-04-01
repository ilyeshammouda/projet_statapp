from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import numpy as np
from classes import image_processing

# Load image and get sparse vector and image shape this methode will return a black and white photo
img_path = 'random.jpg'
coef_vec,coeff_size,coeff_shape= image_processing.image_tansformation_to_spare_vector_BW(img_path)

# Reconstruct image from sparse vector and image shape
img_thresh = image_processing.image_from_sparse_vector_BW(coef_vec,coeff_size,coeff_shape)

# Display original and thresholded images
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
cv2_imshow(img)
cv2_imshow(img_thresh.astype(np.uint8))


#this methode returns a RBG image
coef_vec,slices,coeff_size,coeff_shape =image_processing.compress_image_RGB('random.jpg')
img = image_processing.decompress_photo_RGB(coef_vec,slices,coeff_size,coeff_shape,wt='haar')
plt.imshow(img)
