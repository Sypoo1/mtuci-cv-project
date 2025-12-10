import os
import warnings

import cv2 as cv
import numpy as np
from paddleocr import TextRecognition
from PIL import Image
from ultralytics import YOLO

from utils import display_image, normalize_plate

warnings.filterwarnings("ignore")
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"




def main(folder_path: str):
    car_images = list(
        map(lambda x: os.path.join(folder_path, x), os.listdir(folder_path))
    )

    plate_detection_model = YOLO("tests/best1.pt")

    model = TextRecognition(model_name="PP-OCRv5_server_rec")

    for car_image in car_images:
        file_name = os.path.basename(car_image)
        img = cv.imread(car_image)

        detections = plate_detection_model(img)[0]

        for idx, detection in enumerate(detections.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = detection

            if class_id == 0:
                plate_crop = img[int(y1) : int(y2), int(x1) : int(x2)]

                # plate_crop = preprocess(plate_crop)

                plate_file_path = f"tmp/{idx}_{file_name}"
                cv.imwrite(plate_file_path, plate_crop)

                # OCR
                result = model.predict(plate_file_path)

                predicted_text = ""
                for res in result:
                    res.save_to_img("output")
                    res.save_to_json("output")
                    predicted_text += res["rec_text"]


                predicted_text = normalize_plate(predicted_text)


                cv.rectangle(
                    img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
                )

                if predicted_text:
                    cv.putText(
                        img,
                        predicted_text,
                        (int(x1), int(y1) - 10),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2,
                        cv.LINE_AA,
                    )

                # показываем обновлённое изображение
                display_image(img)
                plate_file_path_done = f"output/{idx}_{file_name}"
                cv.imwrite(plate_file_path_done, img)


def set_image_dpi(img_array):
    im = Image.fromarray(img_array)
    length_x, width_y = im.size
    factor = min(1, float(1024.0 / length_x))
    size = int(factor * length_x), int(factor * width_y)
    im_resized = im.resize(size, Image.Resampling.LANCZOS)


    img_resized = np.array(im_resized)
    return img_resized

def preprocess(img):

    img = set_image_dpi(img)

    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv.normalize(img, norm_img, 0, 255, cv.NORM_MINMAX)
    img = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)

    return img


if __name__ == "__main__":
    main("data")
    # main("custom_data")
