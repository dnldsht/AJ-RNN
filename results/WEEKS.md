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


## Results
### GRU-128
- learnig rate  `1e-6`

  - batch size 32: train 98.7% / val 19% / test 29% (9 epochs)
  - batch size 64: train 97.6% / val 19.9% / test 29.6% (47 epochs)
  - batch size 128: train 92.2% / val 51.6% / test 58.8% (63 epochs)
  - batch size 256: train 87.4% / val 71.9% / test 78.3% (196 epochs)

- learnig rate  `1e-7`

  - batch size 32: train 93.7% / val 27% / test 30% (103 epochs)
  - batch size 64: train 92% / val 29% / test 33% (202 epochs)
  - batch size 128: train 88.5% / val 58.8% / test 69.4% (199 epochs)
  - batch size 256: train 86% / val 74% / test 79.8% (600 epochs)
  - batch size 512: train 85% / val 81% / test 85% (598 epochs)

- learnig rate  `1e-8`
  
  - batch size 128: train  / val / test ( epochs)
  - batch size 256: train  / val / test ( epochs)
  
# Week 48
## Experiments
- lr 1e-6
- bs 32 256
-- plot test metrics each epoc

## details

- GRU128-G5-LR6-BS256
  - seed 89
  - seed 27836
  - seed 196

- GRU128-G5-LR6-BS32
  - seed 89 (ongoing sophia)
  - seed 27836 (TODO)
  - seed 196

- GRU128-G1-LR6-BS256
  - seed 89 (ongoing sophia)
  - seed 27836 (planned sophia)
  - seed 196 (TODO)

- GRU128-G1-LR6-BS32
  - seed 89 (planned sophia)
  - seed 27836
  - seed 196 (TODO)

- GRU128-G1-LR3-BS256
  - seed 89 (planned sophia)
  - seed 27836 (TODO)
  - seed 196

- GRU128-G1-LR3-BS32
  - seed 89 (planned sophia)
  - seed 27836 (TODO)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS256-REG-DROP
  - seed 89
  - seed 27836 (TODO)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS32-REG-DROP
  - seed 89 (planned imac)
  - seed 27836 (skip)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS256-REG
  - seed 89
  - seed 27836 (TODO)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS32-REG
  - seed 89 (ongoing imac)
  - seed 27836 (skip)
  - seed 196 (TODO)


