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

### Batch size 256

- GRU128-G5-LR6-BS256
  - seed 89     maxVal(e:302 [87.3, v:72.0, t:77.6]) maxTrain(e:599 [89.2, v:69.0, t:74.5])
  - seed 27836  maxVal(e:106 [88.0, v:74.4, t:69.2]) maxTrain(e:599 [90.7, v:69.1, t:54.3])
  - seed 196    maxVal(e:89 [85.8, v:76.8, t:76.6]) maxTrain(e:599 [88.1, v:73.1, t:73.6])

- GRU128-G5-LR6-BS256-REG-DROP
  - seed 89     maxVal(e:46 [84.8, v:71.2, t:76.8]) maxTrain(e:599 [89.3, v:63.2, t:69.3])
  - seed 27836  
  - seed 196    maxVal(e:255 [87.0, v:78.1, t:77.3]) maxTrain(e:599 [89.7, v:67.4, t:69.6])

- GRU128-G1-LR6-BS256
  - seed 89     maxVal(e:598 [85.5, v:83.7, t:84.9]) maxTrain(e:599 [85.5, v:83.7, t:84.9])
  - seed 27836  maxVal(e:598 [85.5, v:81.9, t:76.2]) maxTrain(e:599 [85.5, v:81.9, t:76.2])
  - seed 196 (TODO)

- GRU128-G1-LR3-BS256
  - seed 89 (planned sophia)
  - seed 27836 (TODO)
  - seed 196    maxVal(e:599 [97.9, v:87.7, t:82.5]) maxTrain(e:585 [98.0, v:86.5, t:81.8])

- GRU128-G1-LR3-BS256-REG-DROP (best)
  - seed 89     maxVal(e:587 [97.8, v:87.1, t:88.3]) maxTrain(e:593 [99.4, v:85.7, t:87.6])
  - seed 27836 (TODO)
  - seed 196    maxVal(e:577 [85.3, v:84.5, t:82.6]) maxTrain(e:196 [89.8, v:42.1, t:45.7])

- GRU128-G1-LR3-BS256-REG (2 best)
  - seed 89     maxVal(e:584 [96.1, v:83.9, t:76.2]) maxTrain(e:595 [96.7, v:72.8, t:71.8])
  - seed 27836 (TODO)
  - seed 196    maxVal(e:577 [85.3, v:84.5, t:82.6]) maxTrain(e:196 [89.8, v:42.1, t:45.7])

### Batch size 32

- GRU128-G5-LR6-BS32
  - seed 89     maxVal(e:23 [95.1, v:19.1, t:28.8]) maxTrain(e:537 [99.5, v:18.0, t:26.9])
  - seed 27836 (TODO)
  - seed 196    maxVal(e:6 [90.7, v:27.8, t:29.6]) maxTrain(e:599 [99.4, v:16.4, t:16.9])

- GRU128-G5-LR6-BS32-REG-DROP
  - seed 89     maxVal(e:10 [93.0, v:18.9, t:28.9]) maxTrain(e:247 [99.4, v:16.9, t:26.7])(ongoing imac)
  - seed 27836 (TODO)
  - seed 196 (planned imac)

- GRU128-G1-LR4-BS32-REG-DROP
  - seed 89 (planned imac)
  - seed 27836 (TODO)
  - seed 196 (planned imac)

- GRU128-G1-LR6-BS32 (best)
  - seed 89 maxVal(e:11 [82.7, v:61.9, t:68.6]) maxTrain(e:11 [82.7, v:61.9, t:68.6]) (ongoing sophia)
  - seed 27836  maxVal(e:44 [88.2, v:67.4, t:57.7]) maxTrain(e:599 [92.2, v:33.9, t:23.4])
  - seed 196 (planned imac)

- GRU128-G1-LR3-BS32
  - seed 89 (planned sophia)
  - seed 27836 (TODO)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS32-REG-DROP
  - seed 89     maxVal(e:305 [97.8, v:20.4, t:29.3]) maxTrain(e:553 [99.0, v:4.1, t:4.9])
  - seed 27836 (skip)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS32-REG
  - seed 89     maxVal(e:491 [98.2, v:20.6, t:30.1]) maxTrain(e:450 [98.9, v:5.2, t:8.1])
  - seed 27836 (skip)
  - seed 196 (TODO)


