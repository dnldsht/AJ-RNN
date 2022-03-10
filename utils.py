# -*- coding: utf-8 -*-
import numpy as np
import copy
import tensorflow as tf
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


def next_batch(batch_size, data, label, end_to_end, input_dimension_size, num_step, Trainable):
    if end_to_end:
        data[np.where(np.isnan(data))] = MISSING_VALUE
    need_label = copy.deepcopy(label)
    label = convertToOneHot(label, num_classes=len(np.unique(label)))
    assert data.shape[0] == label.shape[0]
    assert data.shape[0] >= batch_size
    row = data.shape[0]
    batch_len = int(row / batch_size)
    left_row = row - batch_len * batch_size

    # shuffle data for train
    if Trainable:
        indices = np.random.permutation(data.shape[0])
        rand_data = data[indices]
        rand_label = label[indices]
        need_rand_label = need_label[indices]
    else:
        rand_data = data
        rand_label = label
        need_rand_label = need_label

    for i in range(batch_len):
        batch_input = rand_data[i*batch_size: (i+1)*batch_size, :]
        batch_prediction_target = rand_data[i *
                                            batch_size: (i+1)*batch_size, input_dimension_size:]
        mask = np.ones_like(batch_prediction_target)
        mask[np.where(batch_prediction_target == MISSING_VALUE)] = 0
        batch_label = rand_label[i*batch_size: (i+1)*batch_size, :]
        batch_need_label = need_rand_label[i*batch_size: (i+1)*batch_size]
        yield (batch_input.reshape(-1, num_step, input_dimension_size), batch_prediction_target.reshape(-1, num_step - 1, input_dimension_size), mask.reshape(-1, num_step - 1, input_dimension_size), batch_label, batch_size, batch_need_label)

    # padding data for equal batch_size
    if left_row != 0:
        need_more = batch_size - left_row
        need_more = np.random.choice(np.arange(row), size=need_more)
        batch_input = np.concatenate(
            (rand_data[-left_row:, :], rand_data[need_more]), axis=0)
        batch_prediction_target = np.concatenate(
            (rand_data[-left_row:, :], rand_data[need_more]), axis=0)[:, input_dimension_size:]
        assert batch_input.shape[0] == batch_prediction_target.shape[0]
        assert batch_input.shape[1] - \
            input_dimension_size == batch_prediction_target.shape[1]
        mask = np.ones_like(batch_prediction_target)
        mask[np.where(batch_prediction_target == MISSING_VALUE)] = 0
        batch_label = np.concatenate(
            (rand_label[-left_row:, :], rand_label[need_more]), axis=0)
        batch_need_label = np.concatenate(
            (need_rand_label[-left_row:], need_rand_label[need_more]), axis=0)
        yield (batch_input.reshape(-1, num_step, input_dimension_size), batch_prediction_target.reshape(-1, num_step - 1, input_dimension_size), mask.reshape(-1, num_step - 1, input_dimension_size), batch_label, left_row, batch_need_label)


def load_dataset(filename, extra=False):
    data, labels = load_data(filename)
    data[np.where(np.isnan(data))] = MISSING_VALUE
    num_steps = data.shape[1]

    labels, num_classes = transfer_labels(labels)
    labels = convertToOneHot(labels, num_classes=len(np.unique(labels)))
    prediction_target = data[:, 1:]
    mask = np.ones_like(prediction_target)
    mask[np.where(prediction_target == MISSING_VALUE)] = 0

    data = data.reshape(-1, num_steps, 1)
    prediction_target = prediction_target.reshape(-1, num_steps - 1, 1)
    mask = mask.reshape(-1, num_steps - 1, 1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (data, prediction_target, mask, labels))

    if extra:
        return dataset, num_classes, num_steps, 1
    return dataset


def get_cl2objs(lut):
    clID_col = lut[:, 1]
    clID = np.unique(clID_col)
    hashClID2obj = {}
    for val in clID:
        idx = np.where(clID_col == val)
        t_labels = lut[idx]
        hashClID2obj[val] = np.unique(t_labels[:, 0])
    return hashClID2obj


def get_data(objs, lut, data, prediction_target, mask, labels):
    objID_col = lut[:, 1]
    tot_data = []
    tot_prediction_target = []
    tot_mask = []
    tot_labels = []

    for obj in objs:
        idx = np.where(objID_col == obj)
        # print(idx)
        tot_data.append(data[idx])
        tot_labels.append(labels[idx])
        #idx = np.add(idx, 1)
        tot_prediction_target.append(prediction_target[idx])
        tot_mask.append(mask[idx])

    tot_data = np.concatenate(tot_data, axis=0)
    tot_prediction_target = np.concatenate(tot_prediction_target, axis=0)
    tot_mask = np.concatenate(tot_mask, axis=0)
    tot_labels = np.concatenate(tot_labels, axis=0)

    return tot_data, tot_prediction_target, tot_mask, tot_labels


def adapt_slices(slices, num_steps, num_bands):
    data, prediction_target, mask, labels = slices
    prediction_target = data[:, 1:]
    masks = masks[:, 1:]
    mask = np.ones_like(prediction_target)
    mask[np.where(masks == 0)] = 0

    data = data.reshape(-1, num_steps, num_bands)
    prediction_target = prediction_target.reshape(-1, num_steps - 1, num_bands)
    mask = mask.reshape(-1, num_steps - 1, num_bands)
    return data, prediction_target, mask, labels


def check_missing_ratio(masks):
    total = np.multiply(*masks.shape)
    ones = np.count_nonzero(masks)
    missing_ratio = 1 - ones/total
    print(f"MISSING RATIO: {missing_ratio}")


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = 137606

    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12, reshuffle_each_iteration=False)

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

    check_missing_ratio(masks)

    num_steps = data.shape[1]
    num_bands = data.shape[2]
    labels, num_classes = transfer_labels(lut[:, 1])
    labels = convertToOneHot(labels, num_classes=len(np.unique(labels)))
    prediction_target = data[:, 1:]
    masks = masks[:, 1:]
    mask = np.ones_like(prediction_target)
    mask[np.where(masks == 0)] = 0

    data = data.reshape(-1, num_steps, num_bands)
    prediction_target = prediction_target.reshape(-1, num_steps - 1, num_bands)
    mask = mask.reshape(-1, num_steps - 1, num_bands)

    dataset = tf.data.Dataset.from_tensor_slices(
        (data, prediction_target, mask, labels))

    training, validation, test = get_dataset_partitions_tf(
        dataset, train_split=0.4, val_split=0.2, test_split=0.4, shuffle=True, shuffle_size=10000)

    return training, validation, test, num_classes, num_steps, num_bands

    #asd = (data, prediction_target, mask, labels)

    hashClID2obj = get_cl2objs(lut)
    train_perc = .4
    train_valid = .3

    training = None
    validation = None
    test = None

    for k in hashClID2obj.keys():
        objIds = hashClID2obj[k]
        objIds = tf.random.shuffle(objIds)
        limit_train = int(len(objIds) * train_perc)
        limit_valid = limit_train + int(len(objIds) * train_valid)

        train_obj = objIds[0:limit_train]
        valid_obj = objIds[limit_train:limit_valid]
        test_obj = objIds[limit_valid::]

        slices = get_data(train_obj, lut, asd)
        slices = adapt_slices(slices, num_steps, num_bands)
        d = tf.data.Dataset.from_tensor_slices(slices)
        if training is None:
            training = d
        else:
            training.concatenate(d)

        slices = get_data(valid_obj, lut, data,
                          prediction_target, mask, labels)
        slices = adapt_slices(slices, num_steps, num_bands)
        d = tf.data.Dataset.from_tensor_slices(slices)
        if validation is None:
            validation = d
        else:
            validation.concatenate(d)

        slices = get_data(test_obj, lut, data,
                          prediction_target, mask, labels)
        slices = adapt_slices(slices, num_steps, num_bands)
        d = tf.data.Dataset.from_tensor_slices(slices)
        if test is None:
            test = d
        else:
            test.concatenate(d)

    # dataset = tf.data.Dataset.from_tensor_slices(
    #     (data, prediction_target, mask, labels))

    return training, validation, test, num_classes, num_steps, num_bands


# def load_sits():
#     data = np.load('SITS-Missing-Data/D1_balaruc_samples.npy')
#     masks = np.load('SITS-Missing-Data/D2_balaruc_masks.npy')
#     lut = np.load('SITS-Missing-Data/D3_balaruc_lut.npy')
#     num_steps = data.shape[1]
#     num_bands = data.shape[2]
#     labels, num_classes = transfer_labels(lut[:, 1])
#     labels = convertToOneHot(labels, num_classes=len(np.unique(labels)))
#     prediction_target = data[:, 1:]
#     masks = masks[:, 1:]
#     mask = np.ones_like(prediction_target)
#     mask[np.where(masks == 0)] = 0

#     dataset = tf.data.Dataset.from_tensor_slices(
#         (data, prediction_target, mask, labels))
#     print(dataset)

#     return dataset, num_classes, num_steps, num_bands
