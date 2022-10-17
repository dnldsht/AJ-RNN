# Test 1

```python
self.g_optimizer = tf.keras.optimizers.Adam(0)
total_G_loss = G_loss
```

## 100 epochs

history: `tests/1-100epochs.txt`

- accuracy: 0.7545
- val_accuracy: 0.7475
- test_accuracy: 0.7744

# Test 2

```python
self.g_optimizer = tf.keras.optimizers.Adam(0)
total_G_loss = G_loss + 1e-4 * regularization_loss
```

### 200 epochs

history: `tests/2-200epochs.txt`

- accuracy: 0.8329
- val_accuracy: 0.7672
- test_accuracy: 0.7713

# Test 3

Commit [eedadc85908f5f4e3dfd8bc4abde8edd5e984ebf](https://github.com/dnldsht/AJ-RNN/commit/eedadc85908f5f4e3dfd8bc4abde8edd5e984ebf)

## 200 epochs

- accuracy: 0.8018
- val_accuracy: 0.81555
- test_accuracy: 0.7491788

### relevant changes

```python
self.g_optimizer = tf.keras.optimizers.Adam(0)
total_G_loss = G_loss + 1e-4 * regularization_loss
```

- classifier update weights outside generator epochs

# Test 4

## 300 epochs

history: `tests/4-300e.txt`

- accuracy: 0.8832
- val_accuracy: 0.6605
- test_accuracy: 0.610

Commit [b6cca3b85c4c6c6457645a867ef3d1d3ce478540](https://github.com/dnldsht/AJ-RNN/commit/b6cca3b85c4c6c6457645a867ef3d1d3ce478540)

### relevant changes

```python
self.g_optimizer = tf.keras.optimizers.Adam(0)
self.classifier_optimizer = tf.keras.optimizers.Adam(1e-3)
total_G_loss = loss_imputation + G_loss + 1e-4 * regularization_loss
```

# Test 5

## 300 epochs

history: `tests/5-300e.txt`

- accuracy: 0.7224
- val_accuracy: 0.7587 (max: 0.81540)
- test_accuracy: 0.6788

Commit [c4378ef54cb4d6b0e3084a64f6aaacd0a2dd2daa](https://github.com/dnldsht/AJ-RNN/commit/c4378ef54cb4d6b0e3084a64f6aaacd0a2dd2daa)

### relevant changes

- batch_size: 256
- logged total_G_loss

```python
self.g_optimizer = tf.keras.optimizers.Adam(1e-7)
```

# Test 6

## 300 epochs

history: `tests/6-300e.txt`

- accuracy: 0.3882
- val_accuracy: 0.3353 (max: 0.73327)
- test_accuracy: 0.28116

Commit [c93fec7f23e3a9bd8375da305e92b988a3f306b1](https://github.com/dnldsht/AJ-RNN/commit/c93fec7f23e3a9bd8375da305e92b988a3f306b1)

### relevant changes

- batch_size: 512 (sophia)

```python
self.g_optimizer = tf.keras.optimizers.Adam(1e-6)
total_G_loss = loss_imputation + G_loss + 1e-4 * regularization_loss
```

# Test 7

## 300 epochs

history: `tests/7-300e.txt`

- accuracy: 0.8120
- val_accuracy: 0.8282
- test_accuracy: 0.77026

Commit [f019726cea4d47b75b541bf2299abfbb7ad4a6be](https://github.com/dnldsht/AJ-RNN/commit/f019726cea4d47b75b541bf2299abfbb7ad4a6be)

### relevant changes

- batch_size: 256

```python
self.g_optimizer = tf.keras.optimizers.Adam(1e-8)
total_G_loss = 1e-4 *loss_imputation + G_loss + 1e-4 * regularization_loss
```

# Test 8

## 300 epochs

history: `tests/8-300e.txt`

- accuracy: 0.8063
- val_accuracy: 0.8221
- test_accuracy: 0.7586

## 300+300 epochs

history: `tests/8.1-600e.txt`

- accuracy:
- val_accuracy:
- test_accuracy:

Commit [872c11b74fb2e9aacb7f956518c5c086c3e36418](https://github.com/dnldsht/AJ-RNN/commit/872c11b74fb2e9aacb7f956518c5c086c3e36418)

### relevant changes

- batch_size: 256

```python
self.g_optimizer = tf.keras.optimizers.Adam(1e-8)
total_G_loss = 1e-3 *loss_imputation + G_loss + 1e-4 * regularization_loss
```
