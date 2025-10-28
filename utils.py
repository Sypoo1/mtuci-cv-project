import cv2 as cv


def displaly_image(name):

    cv.imshow('img', name)
    k = cv.waitKey(0) 
    cv.destroyAllWindows()

    return