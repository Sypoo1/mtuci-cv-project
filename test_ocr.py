import re

import cv2
from paddleocr import PaddleOCR


# Preprocess cropped plate image
def preprocess(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img = cv2.fastNlMeansDenoising(img, h=8)
    img = cv2.equalizeHist(img)
    cv2.imwrite("plate_pre.png", img)
    return "plate_pre.png"


ocr = PaddleOCR(
    lang="ru",
    # use_angle_cls=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

# img_path = preprocess("data/plate_test.png")
# img_path = "data/plate_test.png"
img_path = "ocr_test/A001BO92.png"
img_path = "ocr_test/A001HY116.png"
result = ocr.ocr(img_path)
print(result)
for res in result:
    res.print()
    res.save_to_img("output2")
    res.save_to_json("output2")
