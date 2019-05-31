import numpy as np
import random
import os
import cv2

def to_categorical(y, nb_class):
    y = np.asarray(y, dtype='int32')
    # if not nb_class:
    #     nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_class))
    Y[np.arange(len(y)),y] = 1.
    return Y

def get_random_data(annotation_line, num_class, is_train, root='datasets/CUB_200_2011'):
    '''random preprocessing for real-time data augmentation'''
    line = annotation_line.split()
    mean = [103.939, 116.779, 123.68]
    output_size = (448, 448, 3)
    try:
        img_path = os.path.join(root, line[0])
        img = cv2.imread(img_path)
        gt = int(line[1])
        image = cv2.resize(img, (600, 600), interpolation=cv2.INTER_LINEAR)
        w, h, _ = image.shape
        th, tw, _ = output_size
        if is_train:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
            image = image[i:i+th, j:j+tw, :]
            if random.random() < 0.5:
                image = image[:, ::-1, :]
        else:
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))
            image = image[i:i + th, j:j + tw, :]
        image[..., 0] = image[..., 0] - mean[0]
        image[..., 1] = image[..., 1] - mean[1]
        image[..., 2] = image[..., 2] - mean[2]
        image_data = image
        label = to_categorical([gt], num_class)[0]

    except:
        image_data = np.zeros(output_size, dtype=np.float32)
        label = np.zeros((num_class,), dtype=np.float32)

    return image_data, label

def data_generator(annotation_lines, batch_size, num_class, is_train):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        labels = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, label = get_random_data(annotation_lines[i], num_class=num_class, is_train=is_train)
            image_data.append(image)
            labels.append(label)
            i = (i+1) % n
        image_data = np.array(image_data)
        labels = np.array(labels)
        y_true = labels
        yield image_data, [y_true, y_true, y_true, y_true]

def data_generator_wrapper(annotation_lines, batch_size, num_class=200, is_train=False):
    n = len(annotation_lines)
    annotation_lines = np.array(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, num_class, is_train)
