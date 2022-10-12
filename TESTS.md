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

### relevant changes

```python
self.g_optimizer = tf.keras.optimizers.Adam(0)
total_G_loss = G_loss + 1e-4 * regularization_loss
```

- classifier update weights outside generator epochs

# Test 4

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

Commit [c4378ef54cb4d6b0e3084a64f6aaacd0a2dd2daa](https://github.com/dnldsht/AJ-RNN/commit/c4378ef54cb4d6b0e3084a64f6aaacd0a2dd2daa)

### relevant changes

- batch_size: 256
- logged total_G_loss

```python
self.g_optimizer = tf.keras.optimizers.Adam(1e-7)
```
