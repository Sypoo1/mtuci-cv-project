import easyocr
from ultralytics import YOLO
import cv2 as cv
from utils import displaly_image
import warnings
import pytesseract

warnings.filterwarnings('ignore') 


cars = ['data/1.jpg', 'data/2.jpg', 'data/3.jpg']
vehicles_class_ids = [2, 3, 5, 7]
car_detection_model = YOLO('yolo11n.pt')
reader = easyocr.Reader(['ru'], gpu=False)


for car in cars:

    img = cv.imread(car)

    displaly_image(img)


    detections = car_detection_model(img)[0]
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        
        if class_id in vehicles_class_ids:
            car_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            displaly_image(car_crop)

            car_crop_gray = cv.cvtColor(car_crop, cv.COLOR_BGR2GRAY)

            displaly_image(car_crop_gray)

            # _, car_crop_thresh = cv.threshold(car_crop_gray, 64, 255, cv.THRESH_BINARY_INV)
            
            # displaly_image(car_crop_thresh)

            # detections = reader.readtext(car_crop_gray)

            # for detection in detections:
            #     bbox, text, score = detection

            #     text = text.upper().replace(' ', '')

            #     print(text)
            detection = pytesseract.image_to_string(car_crop_gray, lang='rus')
            print(f'npytesseract: {detection}')

    # break

