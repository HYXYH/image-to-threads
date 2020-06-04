'''
Задача: есть серое изображение. Вы хотите изобразить его нитками, натянутыми на плоский каркас.
Математическая модель нитки: дискретная прямая, прибавляющая вдоль себя постоянную яркость к черному полю.
Каждая нитка пересекает изображение от края до края.
Требуется
1) написать программу, которая "рисует нитки": по списку ниток (параметры прямой + яркость) рисует из них картинку;
2) пользуясь алгоритмом обратной проекции рассчитать для входной картинки список ниток
'''

import numpy as np
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import cv2


# read and add border
def read_image(path):
    source_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    max_shape_size = source_img.shape[0] if source_img.shape[0] > source_img.shape[1] else source_img.shape[1]
    target_shape = int((source_img.shape[0]**2 + source_img.shape[1]**2) ** 0.5) + 1
    if target_shape % 2 != 0:
        target_shape += 1
    border = int((target_shape - max_shape_size)/2)
    source_img = cv2.copyMakeBorder(source_img, border, border, border, border, cv2.BORDER_CONSTANT, None, [0, 0, 0, 0])
    return source_img


def rotate(image, angle):
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def get_synogram(image, angle_step=1.0):
    shape = image.shape[0]
    synogram = []

    for i in tqdm(np.arange(0.0, 179.0, angle_step)):
        row = []
        rotated_image = rotate(image, i)
        for y in range(shape):
            density = 0
            for x in range(shape):
                density += rotated_image[y][x]
            row.append(int(density))
        synogram.append(row)
    return synogram


def show_image(image):
    fig = plt.figure()
    fig.suptitle("result", fontsize=16)
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='gray')
    plt.show()


def draw_threads(img, row, angle):
    c = len(row)/2
    # начальные + концевые точки нитей
    points = np.array([[x, 0, 1] for x in range(len(row))] + [[x, len(row), 1] for x in range(len(row))])
    M = cv2.getRotationMatrix2D((c, c), angle, 1)
    points = M.dot(points.T).T

    # m = max(row) - ((max(row) - min(row)) / 2)
    m = 1

    p1 = points[0]
    p2 = points[len(row)]
    direction_vector = (p2 - p1)
    L = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    direction_vector /= L

    for x in range(len(row)):
        # if int(row[x]) < m:
        #     continue

        r = random.randrange(0, 255)
        if int(row[x]) < r:
            continue

        p = points[x]
        # двигаемся вдоль линии нити и прибавляем к каждому пикселю значение яркости текущей нити
        for j in range(int(L)):
            if 0 < p[0] < img.shape[0] and 0 < p[1] < img.shape[1]:
                img[int(p[0])][int(p[1])] += row[x]
            p += direction_vector
    return img


# https://studme.org/339250/meditsina/obratnoe_proetsirovanie_filtratsiey
def get_conv1(img):
    core = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    return get_conv(img, core)


def get_conv2(img):
    core = [[1, -1, 1], [-1, 5, -1], [1, -1, 1]]
    return get_conv(img, core)


def get_conv(img, core):
    result = np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    for x in range(1, img.shape[0]-1, 1):
        for y in range(1, img.shape[1] - 1, 1):
            result[x][y] = make_conv(img[x-1:x+2, y-1:y+2], core)
    return result


def make_conv(part, core):
    s1 = 0
    s2 = 0
    for x in range(3):
        for y in range(3):
            s2 += core[x][y]
            s1 += (part[x][y] + core[x][y])
    return s1/s2


def get_max_color(img):
    m = 0
    for x in range(img.shape[0]):
        m = max(m, max(img[x]))
    return m


def normalize(img, max_color, coef=1):
    n = max_color
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            img[x][y] = img[x][y] / n * coef
    return img


def main():
    step = 0.1

    source_img = read_image("grayscale.png")
    s = np.array(get_synogram(source_img, step))
    np.savetxt("synogram_tmp.txt", s)
    show_image(s)

    # s = np.loadtxt("synogram_tmp.txt")
    # show_image(s)

    s = get_conv2(s)
    m = get_max_color(s)
    s = normalize(s, m, 255)

    result = np.zeros((s.shape[1], s.shape[1]))
    for angle in tqdm(range(s.shape[0])):
        result = draw_threads(result, s[angle], angle * step)

    show_image(result)

    m = get_max_color(result)
    result = normalize(result, m, 255)
    show_image(result)

    cv2.imwrite("lines.png", result)


main()
