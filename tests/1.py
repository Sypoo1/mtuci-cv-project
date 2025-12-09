from ultralytics import YOLO
import time
import cv2

# Load models
model1 = YOLO("best1.pt")
model2 = YOLO("best2.pt")

# Run inference for each image
for i in range(1, 9):
    # image_path = f"/home/sypoo/Mtuci/mtuci-cv-project/data/{i}.jpg"
    image_path = '/home/sypoo/Mtuci/mtuci-cv-project/ocr_test/A001BO92.png'
    # Run inference with both models
    results1 = model1(image_path)
    results2 = model2(image_path)

    # Display images with model labels
    cv2.imshow(f'Model 1 - Image {i}', results1[0].plot())
    cv2.imshow(f'Model 2 - Image {i}', results2[0].plot())

    # Print model info in console
    print(f"\n{'='*50}")
    print(f"Image: {i}.jpg")
    print(f"Model 1 detected: {len(results1[0].boxes) if results1[0].boxes else 0} objects")
    print(f"Model 2 detected: {len(results2[0].boxes) if results2[0].boxes else 0} objects")

    # Wait for key press to continue
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Small delay between images
    time.sleep(1)