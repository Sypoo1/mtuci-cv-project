import easyocr
import cv2 as cv
from utils import displaly_image
import warnings
import pytesseract
from PIL import Image


warnings.filterwarnings('ignore') 




# cars = ['data/1.jpg', 'data/2.jpg', 'data/3.jpg']
reader = easyocr.Reader(['ru'], gpu=False)

img_path = 'data/plate_test.png'


test = cv.imread(img_path)

plate_crop_gray = cv.cvtColor(test, cv.COLOR_BGR2GRAY)

displaly_image(plate_crop_gray)

_, plate_crop_thresh = cv.threshold(plate_crop_gray, 64, 255, cv.THRESH_BINARY_INV)


detections = reader.readtext(plate_crop_gray)

for detection in detections:
    bbox, text, score = detection

    text = text.upper().replace(' ', '')

    print(f'\neasyocr: {text}')


# image = Image.open('image.png')

text = pytesseract.image_to_string(plate_crop_gray, lang='rus')
text = text.upper().replace(' ', '')

print(f'pytesseract: {text}')