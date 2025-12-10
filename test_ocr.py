from paddleocr import TextRecognition
model = TextRecognition(model_name="PP-OCRv5_server_rec")
output = model.predict(input="tmp/0_1.jpg", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output5/")
    res.save_to_json(save_path="./output5/res.json")
