import easyocr
from ultralytics import YOLO
import cv2 as cv
from utils import displaly_image
import warnings
import pytesseract

from paddleocr import PaddleOCR
warnings.filterwarnings('ignore')


cars = ['data/1.jpg', 'data/2.jpg', 'data/3.jpg']
vehicles_class_ids = [2, 3, 5, 7]
car_detection_model = YOLO('tests/best1.pt')
reader = easyocr.Reader(['ru'], gpu=False)

ocr = PaddleOCR(
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    use_doc_unwarping=False, # Disables text image rectification model via this parameter
    use_textline_orientation=False, # Disables text line orientation classification model via this parameter
    lang="ru"
)


for car in cars:

    img = cv.imread(car)

    displaly_image(img)

    detections = car_detection_model(img)[0]
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection


        if class_id == 0:
            plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            displaly_image(plate_crop)

            plate_crop_gray = cv.cvtColor(plate_crop, cv.COLOR_BGR2GRAY)

            displaly_image(plate_crop_gray)

            _, plate_crop_thresh = cv.threshold(plate_crop_gray, 64, 255, cv.THRESH_BINARY_INV)

            displaly_image(plate_crop_thresh)

            detections = reader.readtext(plate_crop_gray)

            for detection in detections:
                bbox, text, score = detection

                text = text.upper().replace(' ', '')

                print(text)
            detection = pytesseract.image_to_string(plate_crop_gray, lang='rus')
            print(detection)


            result = ocr.predict("data/plate_test.png")
            for res in result:
                res.print()
                res.save_to_img("output")
                res.save_to_json("output")

    break
