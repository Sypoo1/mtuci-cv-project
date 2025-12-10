import cv2 as cv


# def display_image(name):

#     cv.imshow('img', name)
#     k = cv.waitKey(0)
#     cv.destroyAllWindows()

#     return


def display_image(image):
    # Увеличиваем размер изображения в 2 раза (например)
    scaled_image = cv.resize(image, None, fx=3, fy=3, interpolation=cv.INTER_LINEAR)

    # Отображаем изображение
    cv.imshow('img', scaled_image)

    # Ждём нажатия клавиши
    k = cv.waitKey(0)

    # Закрываем все окна
    cv.destroyAllWindows()
    return