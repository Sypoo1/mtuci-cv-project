import os
import warnings

import cv2 as cv
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from ultralytics import YOLO

from utils import display_image

warnings.filterwarnings("ignore")


def main(folder_path: str):

    car_images = list(map(lambda x: os.path.join(folder_path, x), os.listdir(folder_path)))

    plate_detection_model = YOLO("tests/best1.pt")

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="ru",
    )


    for car_image in car_images:
        file_name = os.path.basename(car_image)
        img = cv.imread(car_image)

        # displaly_image(img)

        detections = plate_detection_model(img)[0]
        for idx, detection in enumerate(detections.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = detection

            if class_id == 0:
                plate_crop = img[int(y1) : int(y2), int(x1) : int(x2), :]

                processed_plate_crop = preprocess(plate_crop)

                plate_file_path = f'tmp/{idx}_{file_name}'

                input_img = plate_crop

                cv.imwrite(plate_file_path, input_img)

                # displaly_image(plate_crop)
                # cv.imwrite(f"tmp/crop_{file_name}", plate_crop)



                result = ocr.predict(plate_file_path)
                for res in result:
                    # res.print()
                    res.save_to_img("output")
                    # res.save_to_json("output")

                filename_without_extension, extension = os.path.splitext(file_name)

                output_file_path = f'output/{idx}_{filename_without_extension}_ocr_res_img{extension}'
                # print(output_file_path)

                output = cv.imread(output_file_path)
                display_image(output)



        # break



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