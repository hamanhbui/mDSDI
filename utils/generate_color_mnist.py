import gzip

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


image_size = 28
df_orig_train = pd.read_csv("/home/ubuntu/DSDI_data/MNIST/mnist_train.csv")
df_orig_test = pd.read_csv("/home/ubuntu/DSDI_data/MNIST/mnist_test.csv")

labels_train = df_orig_train["label"]
df_train_digits = df_orig_train.drop("label", axis=1)

labels_test = df_orig_test["label"]
df_test_digits = df_orig_test.drop("label", axis=1)

# Green
maroon3 = [176, 48, 96]
dimgray = [105, 105, 105]
darkolivegreen = [85, 107, 47]
deeppink = [255, 20, 147]
darkslateblue = [72, 61, 139]
darkcyan = [0, 139, 139]
yellowgreen = [154, 205, 50]
darkblue = [0, 0, 139]
purple2 = [127, 0, 127]
darkseagreen = [143, 188, 143]
green_domain = [
    maroon3,
    dimgray,
    darkolivegreen,
    deeppink,
    darkslateblue,
    darkcyan,
    yellowgreen,
    darkblue,
    purple2,
    darkseagreen,
]

# Blue
orange = [255, 165, 0]
yellow = [255, 255, 0]
lawngreen = [124, 252, 0]
springgreen = [0, 255, 127]
crimson = [220, 20, 60]
aqua = [0, 255, 255]
green = [0, 128, 0]
orangered = [255, 69, 0]
coral = [255, 127, 80]
fuchsia = [255, 0, 255]
blue_domain = [orange, yellow, lawngreen, springgreen, crimson, aqua, green, orangered, coral, fuchsia]

# Red
dodgerblue = [30, 144, 255]
khaki = [240, 230, 140]
lightgreen = [144, 238, 144]
saddlebrown = [85, 107, 47]
mediumslateblue = [123, 104, 238]
violet = [238, 130, 238]
lightpink = [255, 182, 193]
lightsteelblue = [176, 196, 222]
blue = [0, 0, 255]
deepskyblue = [0, 191, 255]
red_domain = [
    dodgerblue,
    khaki,
    lightgreen,
    saddlebrown,
    mediumslateblue,
    violet,
    lightpink,
    lightsteelblue,
    blue,
    deepskyblue,
]

# Target - Orange
black = [0, 0, 0]
khaki = [240, 230, 140]
black = [0, 0, 0]
black = [0, 0, 0]
crimson = [220, 20, 60]
black = [0, 0, 0]
lightpink = [255, 182, 193]
black = [0, 0, 0]
black = [0, 0, 0]
darkseagreen = [143, 188, 143]
target_domain = [
    dodgerblue,
    khaki,
    lawngreen,
    saddlebrown,
    crimson,
    darkcyan,
    green,
    lightsteelblue,
    purple2,
    darkseagreen,
]


def add_color(grey_img, c1_img, c2_img, c3_img, color):
    # Blue
    c1_img[grey_img == 0] = color[2]
    # Green
    c2_img[grey_img == 0] = color[1]
    # Red
    c3_img[grey_img == 0] = color[0]
    return c1_img, c2_img, c3_img


# Red train
train_paths = []
train_labels = []

for index, row in df_train_digits.iterrows():
    if index > 999:
        break
    data = df_train_digits.iloc[index].to_numpy()
    label = labels_train[index]

    path = "red/" + str(label) + "/tr_image_" + str(index) + ".png"
    train_paths.append(path)
    train_labels.append(label)

    grey_img = data.reshape(image_size, image_size, 1)

    c1_img = grey_img.copy()
    c2_img = grey_img.copy()
    c3_img = grey_img.copy()

    c1_img[grey_img != 0] = 0
    c2_img[grey_img != 0] = 0
    c3_img[grey_img != 0] = 255

    c1_img, c2_img, c3_img = add_color(grey_img, c1_img, c2_img, c3_img, red_domain[label])

    rgb_img = np.append(c1_img, c2_img, axis=2)
    rgb_img = np.append(rgb_img, c3_img, axis=2)

    rgb_img = rgb_img.astype(np.uint8)

    cv2.imwrite("/home/ubuntu/data/colored_MNIST/Raw images/" + path, rgb_img)

tr_meta_files = pd.DataFrame({"path": train_paths, "label": train_labels})
tr_meta_files.to_csv(
    "/home/ubuntu/data/colored_MNIST/Train val splits/red_train_kfold.txt",
    header=None,
    sep=" ",
    encoding="utf-8",
    index=False,
)

# Green train
train_paths = []
train_labels = []

for index, row in df_train_digits.iterrows():
    if index > 999:
        break
    data = df_train_digits.iloc[index].to_numpy()
    label = labels_train[index]

    path = "green/" + str(label) + "/tr_image_" + str(index) + ".png"
    train_paths.append(path)
    train_labels.append(label)

    grey_img = data.reshape(image_size, image_size, 1)

    c1_img = grey_img.copy()
    c2_img = grey_img.copy()
    c3_img = grey_img.copy()

    c1_img[grey_img != 0] = 0
    c2_img[grey_img != 0] = 255
    c3_img[grey_img != 0] = 0

    c1_img, c2_img, c3_img = add_color(grey_img, c1_img, c2_img, c3_img, green_domain[label])

    rgb_img = np.append(c1_img, c2_img, axis=2)
    rgb_img = np.append(rgb_img, c3_img, axis=2)

    rgb_img = rgb_img.astype(np.uint8)

    cv2.imwrite("/home/ubuntu/data/colored_MNIST/Raw images/" + path, rgb_img)

tr_meta_files = pd.DataFrame({"path": train_paths, "label": train_labels})
tr_meta_files.to_csv(
    "/home/ubuntu/data/colored_MNIST/Train val splits/green_train_kfold.txt",
    header=None,
    sep=" ",
    encoding="utf-8",
    index=False,
)

# Blue train
train_paths = []
train_labels = []

for index, row in df_train_digits.iterrows():
    if index > 999:
        break
    data = df_train_digits.iloc[index].to_numpy()
    label = labels_train[index]

    path = "blue/" + str(label) + "/tr_image_" + str(index) + ".png"
    train_paths.append(path)
    train_labels.append(label)

    grey_img = data.reshape(image_size, image_size, 1)

    c1_img = grey_img.copy()
    c2_img = grey_img.copy()
    c3_img = grey_img.copy()

    c1_img[grey_img != 0] = 255
    c2_img[grey_img != 0] = 0
    c3_img[grey_img != 0] = 0

    c1_img, c2_img, c3_img = add_color(grey_img, c1_img, c2_img, c3_img, blue_domain[label])

    rgb_img = np.append(c1_img, c2_img, axis=2)
    rgb_img = np.append(rgb_img, c3_img, axis=2)

    rgb_img = rgb_img.astype(np.uint8)

    cv2.imwrite("/home/ubuntu/data/colored_MNIST/Raw images/" + path, rgb_img)

tr_meta_files = pd.DataFrame({"path": train_paths, "label": train_labels})
tr_meta_files.to_csv(
    "/home/ubuntu/data/colored_MNIST/Train val splits/blue_train_kfold.txt",
    header=None,
    sep=" ",
    encoding="utf-8",
    index=False,
)

# Target
test_paths = []
test_labels = []
for index, row in df_test_digits.iterrows():
    data = df_test_digits.iloc[index].to_numpy()
    label = labels_test[index]

    path = "target_orange/" + str(label) + "/test_image_" + str(index) + ".png"
    test_paths.append(path)
    test_labels.append(label)

    grey_img = data.reshape(image_size, image_size, 1)

    c1_img = grey_img.copy()
    c2_img = grey_img.copy()
    c3_img = grey_img.copy()

    c1_img[grey_img != 0] = 0
    c2_img[grey_img != 0] = 165
    c3_img[grey_img != 0] = 255

    c1_img, c2_img, c3_img = add_color(grey_img, c1_img, c2_img, c3_img, target_domain[label])

    rgb_img = np.append(c1_img, c2_img, axis=2)
    rgb_img = np.append(rgb_img, c3_img, axis=2)

    rgb_img = rgb_img.astype(np.uint8)

    cv2.imwrite("/home/ubuntu/data/colored_MNIST/Raw images/" + path, rgb_img)

test_meta_files = pd.DataFrame({"path": test_paths, "label": test_labels})
test_meta_files.to_csv(
    "/home/ubuntu/data/colored_MNIST/Train val splits/target_orange_test_kfold.txt",
    header=None,
    sep=" ",
    encoding="utf-8",
    index=False,
)
