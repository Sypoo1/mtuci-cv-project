import cv2 as cv


# def display_image(name):

#     cv.imshow('img', name)
#     k = cv.waitKey(0)
#     cv.destroyAllWindows()

#     return


def display_image(image):
    # Увеличиваем размер изображения в 2 раза ( например)
    scaled_image = cv.resize(image, None, fx=1, fy=1, interpolation=cv.INTER_LINEAR)

    # Отображаем изображение
    cv.imshow('img', scaled_image)

    # Ждём нажатия клавиши
    k = cv.waitKey(0)

    # Закрываем все окна
    cv.destroyAllWindows()
    return

def normalize_plate(raw: str) -> str:
    """
    Нормализует символы только если:
      - символ стоит на правильной позиции (буквенной или цифровой)
      - замена однозначна
      - замена входит в список правил
    Если символ не подходит под правило — возвращается как есть.
    """

    if not raw:
        return ""

    # оставляем только буквы и цифры
    s = "".join([c for c in raw.upper() if c.isalnum()])
    chars = list(s)

    # позиции букв
    letter_positions = {0, 4, 5}
    # позиции цифр
    digit_positions = {1, 2, 3, 6, 7}

    normalized = []

    for i, ch in enumerate(chars):

        # --- БУКВЕННЫЕ позиции ---
        if i in letter_positions:
            if ch == "Y":
                normalized.append("y")
            elif ch == "V":
                normalized.append("y")
            elif ch == "B":
                normalized.append("B")
            elif ch == "0":      # 0 -> О, но только в букве
                normalized.append("O")
            elif ch == "1":
                normalized.append("T")
            else:
                normalized.append(ch)

        # --- ЦИФРОВЫЕ позиции ---
        elif i in digit_positions:
            if ch == "Z":        # Z → 7, но только в цифре
                normalized.append("7")
            else:
                normalized.append(ch)

        # --- Прочие позиции (если номер короткий) ---
        else:
            # никаких замен — символ остаётся в исходном виде
            normalized.append(ch)

    return "".join(normalized)
