# Week 49

## TODO
- light AJRNN
- bs 32 vs 256
- compare interpolation of AJRNN with other interpolation methods 

### BS 32 vs 256

- GRU128-G1-LR3-BS256-REG-DROP (best)
  - seed 89
  - seed 196

- GRU128-G1-LR8-BS32 (best)
  - seed 89
  - seed 196


# light AJRNN
## average

 runs  name                                    accuracy    val_accuracy    test_accuracy    best_val_epoch    G_epoch    hidden_size    batch_size    learning_rate    dropout    train_time
------  -----------------------------------  ----------  --------------  ---------------  ----------------  ---------  -------------  ------------  ---------------  ---------  ------------
     2  G-LIGHT-GRU128-G1-LR6-B32-REG              86.1            49               54.3                94          1            128            32           1e-06         0            2.68
     1  G-LIGHT-GRU128-G1-LR3-B256-REG-DROP        78.6            77.7             78.5                26          1            128           256           0.001         0.5          0.99
     2  G-LIGHT-GRU128-G1-LR5-B32-REG              85.8            40.2             44.2                 6          1            128            32           1e-05         0            1.39
     2  G-LIGHT-GRU128-G1-LR5-B256-REG             85.4            84               83.3               339          1            128           256           1e-05         0            3.37
     2  G-LIGHT-GRU128-G1-LR6-B256-REG             83.7            81.7             85.6              1000          1            128           256           1e-06         0            7.48
     2  G-LIGHT-GRU128-G1-LR7-B32-REG              84.1            51.8             53                   1          1            128            32           1e-07         0            1.4
     2  G-LIGHT-GRU128-G1-LR4-B256-REG             91.5            87               86.6               354          1            128           256           0.0001        0            3.62
     3  G-LIGHT-GRU128-G1-LR8-B32-REG              85.5            56.8             57.1              1000          1            128            32           1e-08         0           13.66
     3  G-LIGHT-GRU128-G1-LR3-B256-REG             80.6            78.2             75.1                77          1            128           256           0.001         0            1.63
     1  G-LIGHT-GRU128-G1-LR3-B32-REG-DROP         98.6            22.1             30.9               169          1            128            32           0.001         0.5          3.56

## raw
name                                     seed    accuracy    val_accuracy  test_accuracy      best_val_epoch    G_epoch    hidden_size    batch_size    learning_rate    dropout  train_time
-----------------------------------  ------  ----------  --------------  ---------------  ----------------  ---------  -------------  ------------  ---------------  ---------  ------------
G-LIGHT-GRU128-G1-LR5-B32-REG           196        84.4            36.2  35.7                            6          1            128            32           1e-05         0    1.39
G-LIGHT-GRU128-G1-LR5-B32-REG            89        87.1            44.1  52.7                           15          1            128            32           1e-05         0    1.51
G-LIGHT-GRU128-G1-LR3-B32-REG-DROP       89        98.6            22.1  30.9                          169          1            128            32           1e-03         0.5  3.56
G-LIGHT-GRU128-G1-LR6-B32-REG           196        86.1            43.2  44.6                           94          1            128            32           1e-06         0    2.68
G-LIGHT-GRU128-G1-LR6-B32-REG            89        86.1            54.9  64.0                           85          1            128            32           1e-06         0    2.37
G-LIGHT-GRU128-G1-LR7-B32-REG           196        78.6            48    42.2                            1          1            128            32           1e-07         0    1.4
G-LIGHT-GRU128-G1-LR7-B32-REG            89        89.6            55.5  63.7                          997          1            128            32           1e-07         0    12.58
G-LIGHT-GRU128-G1-LR8-B32-REG            23        88.9            58.4  54.3                         1000          1            128            32           1e-08         0    13.66
G-LIGHT-GRU128-G1-LR8-B32-REG           196        78.9            52    48.3                            1          1            128            32           1e-08         0    1.44
G-LIGHT-GRU128-G1-LR8-B32-REG            89        88.7            60.1  68.6                          995          1            128            32           1e-08         0    12.14

G-LIGHT-GRU128-G1-LR5-B256-REG          196        86.2            85.1  81.3                          339          1            128           256           1e-05         0    3.37
G-LIGHT-GRU128-G1-LR5-B256-REG           89        84.6            83    85.3                          206          1            128           256           1e-05         0    2.35
G-LIGHT-GRU128-G1-LR4-B256-REG          196        93              87.4  86.9                          354          1            128           256           1e-04         0    3.62
G-LIGHT-GRU128-G1-LR4-B256-REG           89        89.9            86.6  86.3                          352          1            128           256           1e-04         0    3.99
G-LIGHT-GRU128-G1-LR3-B256-REG-DROP      89        78.6            77.7  78.5                           26          1            128           256           1e-03         0.5  0.99
G-LIGHT-GRU128-G1-LR3-B256-REG           23        76.8            75.9  67.4                           77          1            128           256           1e-03         0    1.63
G-LIGHT-GRU128-G1-LR3-B256-REG          196        79.6            79.3  77.6                           99          1            128           256           1e-03         0    1.75
G-LIGHT-GRU128-G1-LR3-B256-REG           89        85.3            79.3  80.4                          276          1            128           256           1e-03         0    3.69
G-LIGHT-GRU128-G1-LR6-B256-REG           89        85.7            83.6  85.6                         1000          1            128           256           1e-06         0    7.48
G-LIGHT-GRU128-G1-LR6-B256-REG          196        81.3            79                                  160          1            128           256           1e-06         0


### BS 32 vs 256

## average
name                            runs    accuracy    val_accuracy    test_accuracy    best_val_epoch    G_epoch    hidden_size    batch_size    learning_rate    dropout    train_time
----------------------------  ------  ----------  --------------  ---------------  ----------------  ---------  -------------  ------------  ---------------  ---------  ------------
GRU128-G1-LR8-BS32                 3        86.6            70.8             71               544.7          1            128            32            1e-08        0           20.91
GRU128-G1-LR3-BS256-REG-DROP       2        89.4            84.8             81.2             229.5          1            128           256            0.001        0.5         13.44


## raw
name                          week      seed    accuracy    val_accuracy  test_accuracy      best_val_epoch    G_epoch    hidden_size    batch_size    learning_rate    dropout  train_time
----------------------------  ------  ------  ----------  --------------  ---------------  ----------------  ---------  -------------  ------------  ---------------  ---------  ------------
GRU128-G1-LR8-BS32            week49      89        85              63.9  71.0                           99          1            128            32            1e-08        0    20.91
GRU128-G1-LR8-BS32            week49      23        87.9            76.5                                692          1            128            32            1e-08        0
GRU128-G1-LR8-BS32            week49     196        86.9            72.1                                840          1            128            32            1e-08        0
GRU128-G1-LR3-BS256-REG-DROP  week49      89        98.3            87.2  85.5                          305          1            128           256            0.001        0.5  13.44
GRU128-G1-LR3-BS256-REG-DROP  week49      23        80.6            82.5  76.8                          154          1            128           256            0.001        0.5  8.79
