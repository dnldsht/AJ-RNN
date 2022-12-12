# tempCNN

https://www.mdpi.com/2072-4292/11/5/523

## Dataset

i've created train and test dataset as follows:

```python
data = np.load('D1_balaruc_samples.npy')
lut = np.load('D3_balaruc_lut.npy')

train_idx, test_idx = get_split_idx(lut)

labels = lut[:,1]
object_ids = lut[:,0]

n, b, c = data.shape
data = data.reshape((n, b*c))

X = np.column_stack((labels, object_ids, data))

X_train = X[train_idx]
X_test = X[test_idx]

np.savetxt("train_dataset.csv", X_train, delimiter=",")
np.savetxt("test_dataset.csv", X_test, delimiter=",")
```

## Training

### Architecture complexity

- epochs: 50, batch_size: 256
- Overall accuracy (OA): 0.947581946849823
- Train loss: 0.17745892703533173
- Training time (s): 935.57
- Test time (s): 3.55

```python
Model: "Archi_3CONV32_1FC256"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 54, 16)]          0

 conv1d (Conv1D)             (None, 54, 32)            2592

 batch_normalization (BatchN  (None, 54, 32)           128
 ormalization)

 activation (Activation)     (None, 54, 32)            0

 dropout (Dropout)           (None, 54, 32)            0

 conv1d_1 (Conv1D)           (None, 54, 32)            5152

 batch_normalization_1 (Batc  (None, 54, 32)           128
 hNormalization)

 activation_1 (Activation)   (None, 54, 32)            0

 dropout_1 (Dropout)         (None, 54, 32)            0

 conv1d_2 (Conv1D)           (None, 54, 32)            5152

 batch_normalization_2 (Batc  (None, 54, 32)           128
 hNormalization)

 activation_2 (Activation)   (None, 54, 32)            0

 dropout_2 (Dropout)         (None, 54, 32)            0

 flatten (Flatten)           (None, 1728)              0

 dense (Dense)               (None, 256)               442624

 batch_normalization_3 (Batc  (None, 256)              1024
 hNormalization)

 activation_3 (Activation)   (None, 256)               0

 dropout_3 (Dropout)         (None, 256)               0

 dense_1 (Dense)             (None, 8)                 2056

=================================================================
Total params: 458,984
Trainable params: 458,280
Non-trainable params: 704
_________________________________________________________________
```

### Architecture depth

```python
Model: "Archi_1CONV256_1FC64"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_1 (InputLayer)        [(None, 54, 16)]          0

 conv1d (Conv1D)             (None, 54, 256)           20736

 batch_normalization (BatchN  (None, 54, 256)          1024
 ormalization)

 activation (Activation)     (None, 54, 256)           0

 dropout (Dropout)           (None, 54, 256)           0

 flatten (Flatten)           (None, 13824)             0

 dense (Dense)               (None, 64)                884800

 batch_normalization_1 (Batc  (None, 64)               256
 hNormalization)

 activation_1 (Activation)   (None, 64)                0

 dropout_1 (Dropout)         (None, 64)                0

 dense_1 (Dense)             (None, 8)                 520

=================================================================
Total params: 907,336
Trainable params: 906,696
Non-trainable params: 640
_________________________________________________________________
```
