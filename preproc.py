import cv2 as cv
import numpy as np

img_path = "tmp/crop_1.jpg"
img = cv.imread(img_path)

norm_img = np.zeros((img.shape[0], img.shape[1]))
img = cv.normalize(img, norm_img, 0, 255, cv.NORM_MINMAX)


img_output_path = "tmp/crop_norm_1.jpg"
cv.imwrite(img_output_path, img)

def remove_noise(image):

    return cv.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 15)


img_output_path = "tmp/crop_noise_1.jpg"
cv.imwrite(img_output_path, remove_noise(img))


