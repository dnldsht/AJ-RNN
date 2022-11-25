# Week 47

## Changes

```python
total_G_loss = loss_imputation + G_loss + regularization_loss + loss_classification

# for-each g_epoch
## update generator
g_grads = tape.gradient(total_G_loss, self.generator.trainable_variables)
self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

## update classifier
c_grads = tape_c.gradient(loss_classification, self.classifier.trainable_variables)
self.classifier_optimizer.apply_gradients(zip(c_grads, self.classifier.trainable_variables))
```

## Experiments

- batch size
- learning rate
