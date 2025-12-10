import re
import os
import cv2
from paddleocr import PaddleOCR


ocr = PaddleOCR(
    lang="ru",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

folder_path = "tmp"

files = list(map(lambda x: os.path.join(folder_path, x), os.listdir(folder_path)))
for f in files:

    result = ocr.predict(f)

    for res in result:
        res.print()
        res.save_to_img("output3")
