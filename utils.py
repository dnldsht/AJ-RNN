# -*- coding: utf-8 -*-
import numpy as np
import copy
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

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


def convert_to_one_hot(vector, num_classes=None):
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
    labels = convert_to_one_hot(labels, num_classes=len(np.unique(labels)))
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

def split_sits(data, prediction_target, mask, labels, num_classes, train_size=0.6, val_size=0.2, test_size=0.2):
    if train_size + val_size + test_size != 1:
        raise ValueError('train_size + val_size + test_size != 1')
    
    split = train_test_split(data, prediction_target, mask, labels, test_size=val_size+test_size, stratify=labels)
    train_data, rest_data, train_target, rest_target, train_mask, rest_mask, train_labels, rest_labels = split

    split2 = train_test_split(rest_data, rest_target, rest_mask, rest_labels, test_size=(test_size+train_size/2), stratify=rest_labels)
    val_data, test_data, val_target, test_target, val_mask, test_mask, val_labels, test_labels = split2

    train_labels, val_labels, test_labels = convert_to_one_hot(train_labels, num_classes), convert_to_one_hot(val_labels, num_classes), convert_to_one_hot(test_labels, num_classes)
    

    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_target, train_mask, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_target, val_mask, val_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_target, test_mask, test_labels))

    return train_dataset, val_dataset, test_dataset

def get_unique_ids_by_class(lut):
    labels = lut[:, 1]
    unique_labels = np.unique(labels)
    ids_by_class = {}
    for label in unique_labels:
      idx = np.where(labels == label)
      lut_subset = lut[idx]
      ids_by_class[label] = np.unique(lut_subset[:, 0])
    return ids_by_class

def get_idx_of_object_ids(ids, lut):
    lut_ids = lut[:, 0]
    tot_idx = []
    for i in ids:
        tot_idx.append(np.where(lut_ids == i)[0])
    tot_idx = np.concatenate(tot_idx, axis=0)
    return tot_idx

def get_split_idx(lut, train_perc=.6, val_perc=.2, test_perc=.2):
    train_idx, valid_idx, test_idx = [], [], []
    unique_ids_by_class = get_unique_ids_by_class(lut)

    for label in unique_ids_by_class:
        ids = unique_ids_by_class[label]
        ids = shuffle(ids, random_state=23)
        
        limit_train = int(len(ids)* train_perc )
        limit_val = limit_train + int(len(ids)* val_perc)
        limit_test = limit_val + int(len(ids)* test_perc)
        
        
        train_idx.extend(get_idx_of_object_ids(ids[0:limit_train], lut))
        valid_idx.extend(get_idx_of_object_ids(ids[limit_train:limit_val], lut))
        test_idx.extend(get_idx_of_object_ids(ids[limit_val:], lut))

    return (train_idx,), (valid_idx,), (test_idx,)

def generate_dataset(idx, data, prediction_target, mask, labels, num_classes):
    data_subset = data[idx]
    mask_subset = mask[idx]
    labels_subset = convert_to_one_hot(labels[idx], num_classes)
    prediction_target_subset = prediction_target[idx]
    return tf.data.Dataset.from_tensor_slices((data_subset, prediction_target_subset, mask_subset, labels_subset ))


def load_sits():
    data = np.load('SITS-Missing-Data/D1_balaruc_samples.npy')
    masks = np.load('SITS-Missing-Data/D2_balaruc_masks.npy')
    lut = np.load('SITS-Missing-Data/D3_balaruc_lut.npy')
    
    data[np.where(masks == 1)] = MISSING_VALUE

    num_steps = data.shape[1]
    num_bands = data.shape[2]
    labels, num_classes = transfer_labels(lut[:, 1])
    # labels = convert_to_one_hot(labels, num_classes=len(np.unique(labels)))
    prediction_target = data[:, 1:]
    mask = np.ones_like(prediction_target)
    mask[np.where(prediction_target == MISSING_VALUE)] = 0

    check_missing_ratio(np.array(mask))

    data = data.reshape(-1, num_steps, num_bands)
    prediction_target = prediction_target.reshape(-1, num_steps - 1, num_bands)
    mask = mask.reshape(-1, num_steps - 1, num_bands)


    # train 0.6, val 0.2, test 0.2
    # train_dataset, val_dataset, test_dataset = split_sits(data, prediction_target, mask, labels, num_classes, train_size=0.6, val_size=0.2, test_size=0.2)
    train_idx, val_idx, test_idx = get_split_idx(lut)
    train_dataset = generate_dataset(train_idx, data, prediction_target, mask, labels, num_classes)
    val_dataset = generate_dataset(val_idx, data, prediction_target, mask, labels, num_classes)
    test_dataset = generate_dataset(test_idx, data, prediction_target, mask, labels, num_classes)
    #print('train:', train_dataset.shape, 'val:', val_dataset.shape, 'test:', test_dataset.shape)
    #print(train_dataset)


    return train_dataset, val_dataset, test_dataset, num_classes, num_steps, num_bands


    

def load(train, test):
    if train == 'SITS':
        return load_sits()
    
    train_dataset, num_classes, num_steps, num_bands = load_dataset(train, True)
    v_dataset = load_dataset(test, False)
    return train_dataset, v_dataset, None, num_classes, num_steps, num_bands
    