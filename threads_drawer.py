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

    pbar = tqdm(np.arange(0.0, 179.0, angle_step))
    pbar.set_description("synogram")
    for i in pbar:
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


def draw_threads(img, row, angle, rand_lines=False):
    c = len(row)/2
    # начальные + концевые точки нитей
    points = np.array([[x, 0, 1] for x in range(len(row))] + [[x, len(row), 1] for x in range(len(row))])
    M = cv2.getRotationMatrix2D((c, c), angle, 1)
    points = M.dot(points.T).T

    p1 = points[0]
    p2 = points[len(row)]
    direction_vector = (p2 - p1)
    L = np.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
    direction_vector /= L

    for x in range(len(row)):
        if rand_lines:
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

# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4341983/
def get_ramp_conv(img):
    result = []
    f = get_filter(img.shape[1])
    pbar = tqdm(range(img.shape[0]))
    pbar.set_description("filter")
    for x in pbar:
        result.append(conv_row(img[x].tolist(), f))
    return np.array(result)


# дискретный вариант, после обратного преобразования Фурье. По ссылке выше есть вывод формулы.
def ramp(n):
    if n == 0:
        return 0.25
    if n % 2 == 0:
        return 0
    return -1 / ((np.pi * n)**2)


# чем длиннее фильтр, тем точнее результат
def get_filter(tail_len):
    ramp_row = [ramp(i) for i in range(tail_len, 0, -1)] + [ramp(0)] + [ramp(i) for i in range(1, tail_len+1, 1)]
    return ramp_row


def conv_row(row, filter):
    tail_len = int((len(filter)-1) / 2)
    row2 = [0] * tail_len + row + [0] * tail_len
    conv = []
    for i in range(len(row)):
        s = 0
        for j in range(len(filter)):
           s += row2[i+j] * filter[j]
        conv.append(s)
    return conv


def get_min_max_color(img):
    m1 = 0
    m2 = 0
    for x in range(img.shape[0]):
        m1 = min(m1, min(img[x]))
        m2 = max(m2, max(img[x]))
    return m1, m2


def normalize(img, coef=1):
    normalized = np.zeros((img.shape[0], img.shape[1]))
    m1, m2 = get_min_max_color(img)
    lift = 0
    if m1 < 0:
        lift = -m1
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            normalized[x][y] = ((img[x][y] + lift) / (m2+lift)) * coef
    return normalized


def main():
    step = 1
    rand_lines = True

    source_img = read_image("grayscale.png")
    s = np.array(get_synogram(source_img, step), dtype=np.float64)
    show_image(s)
    cv2.imwrite("synogram.png", normalize(s, 255))

    np.savetxt("synogram.txt", s)
    # s = np.loadtxt("synogram.txt")
    # show_image(s)

    s = get_ramp_conv(s)
    show_image(s)

    cv2.imwrite("synogram_ramp_filtered.png", normalize(s, 255))

    result = np.zeros((s.shape[1], s.shape[1]), dtype=np.float64)
    pbar = tqdm(range(s.shape[0]))
    pbar.set_description("draw")
    for angle in pbar:
        result = draw_threads(result, s[angle], angle * step, rand_lines)
    show_image(result)
    rl = "randlines" if rand_lines else ""
    cv2.imwrite(f"lines_step{step}_{rl}.png", normalize(result, 255))


main()
