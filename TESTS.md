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

- accuracy: 0.8133
- val_accuracy: 0.828
- test_accuracy: 0.7748

Commit [872c11b74fb2e9aacb7f956518c5c086c3e36418](https://github.com/dnldsht/AJ-RNN/commit/872c11b74fb2e9aacb7f956518c5c086c3e36418)

### relevant changes

- batch_size: 256

```python
self.g_optimizer = tf.keras.optimizers.Adam(1e-8)
total_G_loss = 1e-3 * loss_imputation + G_loss + 1e-4 * regularization_loss
```

# Test 9

NOTE: redo test and log learning rate

Commit [737adbb9a7e038437ef717b72061a3a393066049](https://github.com/dnldsht/AJ-RNN/commit/737adbb9a7e038437ef717b72061a3a393066049)

## 300 epochs

history: `tests/9-300e.txt`

- accuracy: 0.3882
- val_accuracy: 0.3353
- test_accuracy: 0.281

### relevant changes

- batch_size: 256

```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
              initial_learning_rate=1e-5,
              decay_steps=config.batches * config.G_epoch,
              decay_rate=0.97,
              staircase=True)

self.g_optimizer = tf.keras.optimizers.Adam(lr_schedule)


total_G_loss = 1e-3 * loss_imputation + G_loss + 1e-4 * regularization_loss

```

# Test 10

## 300 epochs

history: `tests/10-300e.txt`

- accuracy:
- val_accuracy:
- test_accuracy:

Commit [b02628c15dabed2561cf396eb8c492f0a0788679](https://github.com/dnldsht/AJ-RNN/commit/b02628c15dabed2561cf396eb8c492f0a0788679)

### relevant changes

- batch_size: 256

```python
self.g_optimizer = tf.keras.optimizers.Adam(1e-8)
total_G_loss = 1e-3 * loss_imputation + G_loss + 1e-4 * regularization_loss

# cell-type = LSTM
```

# Test 11

## 600 epochs

history: `tests/11-600e.txt`

- accuracy: 0.8079
- val_accuracy: 0.8310
- test_accuracy: 0.779

Commit [2a8f36baeb26fcf6e4acd87e4d3de8c5d387008e](https://github.com/dnldsht/AJ-RNN/commit/2a8f36baeb26fcf6e4acd87e4d3de8c5d387008e)

### relevant changes

- batch_size: 256

```python
# Discrinator
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_M, labels=M))

# Generator
self.g_optimizer = tf.keras.optimizers.Adam(1e-8)
loss_imputation = tf.reduce_mean(tf.square( (prediction_targets - prediction) * masks )) / (self.config.batch_size)
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predict_M, labels=1 - M) * (1-M))
total_G_loss = loss_imputation + G_loss + regularization_loss
```