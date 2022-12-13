# Week 48

# conclusions
- with BS256
  - G5 & LR6 -> REG & DROP (does not help)
  - with LR6 -> G1 > G5
  - with G1 -> LR3, REG, DROP (best)

- with BS32
  - G5 & LR6 -> REG & DROP (does not help)
  - with G1 -> LR6 (best) try[REG, DROP?]

# TODO
- test different lr
- test different G epochs
- plot test metrics each epoch

## average BS256

### GRU128-G5-LR6-BS256 (3 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 166 | 87.0 | 74.4 | 74.5 |

### GRU128-G5-LR6-BS256-REG-DROP (2 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 151 | 85.9 | 74.7 | 77.0 |

### GRU128-G1-LR6-BS256 (2 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 598 | 85.5 | 82.8 | 80.6 |

### GRU128-G1-LR3-BS256 (1 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 599 | 97.9 | 87.7 | 82.5 |

### GRU128-G1-LR3-BS256-REG-DROP (best) (2 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 582 | 91.5 | 85.8 | 85.4 |

### GRU128-G1-LR3-BS256-REG (2 best) (2 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 581 | 90.7 | 84.2 | 79.4 |

## average BS32

### GRU128-G5-LR6-BS32 (2 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 15 | 92.9 | 23.5 | 29.2 |

### GRU128-G5-LR6-BS32-REG-DROP (1 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 10 | 93.0 | 18.9 | 28.9 |

### GRU128-G1-LR6-BS32 (best) (2 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 28 | 85.5 | 64.7 | 63.1 |

### GRU128-G1-LR3-BS32-REG-DROP (1 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 305 | 97.8 | 20.4 | 29.3 |

### GRU128-G1-LR3-BS32-REG (1 runs)
| type | epochs | train | val | test |
| --- | --- | --- | --- | --- |
| max val | 491 | 98.2 | 20.6 | 30.1 |

## details

### Batch size 256

- GRU128-G5-LR6-BS256
  - seed 89     maxVal(e:302 [87.3, v:72.0, t:77.6])  maxTrain(e:599 [89.2, v:69.0, t:74.5])
  - seed 27836  maxVal(e:106 [88.0, v:74.4, t:69.2])  maxTrain(e:599 [90.7, v:69.1, t:54.3])
  - seed 196    maxVal(e:89 [85.8, v:76.8, t:76.6])   maxTrain(e:599 [88.1, v:73.1, t:73.6])

- GRU128-G5-LR6-BS256-REG-DROP
  - seed 89     maxVal(e:46 [84.8, v:71.2, t:76.8])   maxTrain(e:599 [89.3, v:63.2, t:69.3])
  - seed 27836  
  - seed 196    maxVal(e:255 [87.0, v:78.1, t:77.3])  maxTrain(e:599 [89.7, v:67.4, t:69.6])

- GRU128-G1-LR6-BS256
  - seed 89     maxVal(e:598 [85.5, v:83.7, t:84.9])  maxTrain(e:599 [85.5, v:83.7, t:84.9])
  - seed 27836  maxVal(e:598 [85.5, v:81.9, t:76.2])  maxTrain(e:599 [85.5, v:81.9, t:76.2])
  - seed 196 (TODO)

- GRU128-G1-LR3-BS256
  - seed 89 (planned sophia)
  - seed 27836 (TODO)
  - seed 196    maxVal(e:599 [97.9, v:87.7, t:82.5])  maxTrain(e:585 [98.0, v:86.5, t:81.8])

- GRU128-G1-LR3-BS256-REG-DROP (best)
  - seed 89     maxVal(e:587 [97.8, v:87.1, t:88.3])  maxTrain(e:593 [99.4, v:85.7, t:87.6])
  - seed 27836 (TODO)
  - seed 196    maxVal(e:577 [85.3, v:84.5, t:82.6])  maxTrain(e:196 [89.8, v:42.1, t:45.7])

- GRU128-G1-LR3-BS256-REG (2 best)
  - seed 89     maxVal(e:584 [96.1, v:83.9, t:76.2])  maxTrain(e:595 [96.7, v:72.8, t:71.8])
  - seed 27836 (TODO)
  - seed 196    maxVal(e:577 [85.3, v:84.5, t:82.6])  maxTrain(e:196 [89.8, v:42.1, t:45.7])

### Batch size 32

- GRU128-G5-LR6-BS32
  - seed 89     maxVal(e:23 [95.1, v:19.1, t:28.8])   maxTrain(e:537 [99.5, v:18.0, t:26.9])
  - seed 27836 (TODO)
  - seed 196    maxVal(e:6 [90.7, v:27.8, t:29.6])    maxTrain(e:599 [99.4, v:16.4, t:16.9])

- GRU128-G5-LR6-BS32-REG-DROP
  - seed 89     maxVal(e:10 [93.0, v:18.9, t:28.9])   maxTrain(e:247 [99.4, v:16.9, t:26.7])(ongoing imac)
  - seed 27836 (TODO)
  - seed 196 (planned imac)

- GRU128-G1-LR4-BS32-REG-DROP
  - seed 89 (planned imac)
  - seed 27836 (TODO)
  - seed 196 (planned imac)

- GRU128-G1-LR6-BS32 (best)
  - seed 89 maxVal(e:11 [82.7, v:61.9, t:68.6])       maxTrain(e:11 [82.7, v:61.9, t:68.6]) (ongoing sophia)
  - seed 27836  maxVal(e:44 [88.2, v:67.4, t:57.7])   maxTrain(e:599 [92.2, v:33.9, t:23.4])
  - seed 196 (planned imac)

- GRU128-G1-LR3-BS32
  - seed 89 (planned sophia)
  - seed 27836 (TODO)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS32-REG-DROP
  - seed 89     maxVal(e:305 [97.8, v:20.4, t:29.3])  maxTrain(e:553 [99.0, v:4.1, t:4.9])
  - seed 27836 (skip)
  - seed 196 (TODO)

- GRU128-G1-LR3-BS32-REG
  - seed 89     maxVal(e:491 [98.2, v:20.6, t:30.1])  maxTrain(e:450 [98.9, v:5.2, t:8.1])
  - seed 27836 (skip)
  - seed 196 (TODO)


