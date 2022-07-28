# -*- coding: utf-8 -*-
import numpy as np
import copy
import tensorflow as tf
from sklearn.model_selection import train_test_split

MISSING_VALUE = 128.0


def transfer_labels(labels):
    # some labels are [1,2,4,11,13] and is transfer to standard label format [0,1,2,3,4]
    indexes = np.unique(labels)
    num_classes = indexes.shape[0]
    num_samples = labels.shape[0]

    for i in range(num_samples):
        new_label = np.argwhere(labels[i] == indexes)[0][0]
        labels[i] = new_label
    return labels, num_classes


def load_data(filename):
    data_label = np.loadtxt(filename, delimiter=',')
    data = data_label[:, 1:].astype(np.float32)
    label = data_label[:, 0].astype(np.int32)
    return data, label


def convertToOneHot(vector, num_classes=None):
    # convert label to one_hot format
    vector = np.array(vector, dtype=int)
    if 0 not in np.unique(vector):
        vector = vector - 1
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    assert num_classes is not None

    assert num_classes > 0
    vector = vector % num_classes

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(np.int32)



def load_dataset(filename, extra=False):
    data, labels = load_data(filename)
    data[np.where(np.isnan(data))] = MISSING_VALUE
    num_steps = data.shape[1]

    labels, num_classes = transfer_labels(labels)
    labels = convertToOneHot(labels, num_classes=len(np.unique(labels)))
    prediction_target = data[:, 1:]
    mask = np.ones_like(prediction_target)
    mask[np.where(prediction_target == MISSING_VALUE)] = 0
    check_missing_ratio(mask)

    data = data.reshape(-1, num_steps, 1)
    prediction_target = prediction_target.reshape(-1, num_steps - 1, 1)
    mask = mask.reshape(-1, num_steps - 1, 1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (data, prediction_target, mask, labels))

    if extra:
        return dataset, num_classes, num_steps, 1
    return dataset


def check_missing_ratio(masks):
    total = np.prod(masks.shape)
    ones = np.count_nonzero(masks)
    missing_ratio = 1 - ones/total
    print(f"MISSING RATIO: {missing_ratio:.3f}")


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = 137606

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, reshuffle_each_iteration=False)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


def load_sits():
    data = np.load('SITS-Missing-Data/D1_balaruc_samples.npy')
    masks = np.load('SITS-Missing-Data/D2_balaruc_masks.npy')
    lut = np.load('SITS-Missing-Data/D3_balaruc_lut.npy')
    
    data[np.where(masks == 1)] = MISSING_VALUE

    num_steps = data.shape[1]
    num_bands = data.shape[2]
    labels, num_classes = transfer_labels(lut[:, 1])
    # labels = convertToOneHot(labels, num_classes=len(np.unique(labels)))
    prediction_target = data[:, 1:]
    mask = np.ones_like(prediction_target)
    mask[np.where(prediction_target == MISSING_VALUE)] = 0

    check_missing_ratio(np.array(mask))

    data = data.reshape(-1, num_steps, num_bands)
    prediction_target = prediction_target.reshape(-1, num_steps - 1, num_bands)
    mask = mask.reshape(-1, num_steps - 1, num_bands)


    split = train_test_split(data, prediction_target, mask, labels, test_size=0.2, random_state=42, stratify=labels)
    train_data, test_data, train_target, test_target, train_mask, test_mask, train_labels, test_labels = split
    train_labels, test_labels = convertToOneHot(train_labels, num_classes=num_classes), convertToOneHot(test_labels, num_classes=num_classes)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_target, train_mask, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_target, test_mask, test_labels))
    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (data, prediction_target, mask, labels))

    # training, validation, test = get_dataset_partitions_tf(
    #     dataset, train_split=0.5, val_split=0.25, test_split=0.25, shuffle=True, shuffle_size=10000)

    return train_dataset, test_dataset, num_classes, num_steps, num_bands


def load(train, test):
    if train == 'SITS':
        return load_sits()
    
    train_dataset, num_classes, num_steps, num_bands = load_dataset(train, True)
    v_dataset = load_dataset(test, False)
    return train_dataset, v_dataset, num_classes, num_steps, num_bands
    