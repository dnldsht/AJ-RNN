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
