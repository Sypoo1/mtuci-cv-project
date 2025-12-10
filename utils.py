import cv2 as cv


# def display_image(name):

#     cv.imshow('img', name)
#     k = cv.waitKey(0)
#     cv.destroyAllWindows()

#     return


def display_image(image):
    scaled_image = cv.resize(image, None, fx=1, fy=1, interpolation=cv.INTER_LINEAR)

    cv.imshow('img', scaled_image)

    k = cv.waitKey(0)

    cv.destroyAllWindows()
    return

def normalize_plate(raw: str) -> str:


    if not raw:
        return ""

    s = "".join([c for c in raw.upper() if c.isalnum()])
    chars = list(s)

    letter_positions = {0, 4, 5}
    digit_positions = {1, 2, 3, 6, 7}

    normalized = []

    for i, ch in enumerate(chars):


        if i in letter_positions:
            if ch == "Y":
                normalized.append("y")
            elif ch == "V":
                normalized.append("y")
            elif ch == "B":
                normalized.append("B")
            elif ch == "0":
                normalized.append("O")
            elif ch == "1":
                normalized.append("T")
            else:
                normalized.append(ch)

        elif i in digit_positions:
            if ch == "Z":
                normalized.append("7")
            else:
                normalized.append(ch)

        else:
            normalized.append(ch)

    return "".join(normalized)
