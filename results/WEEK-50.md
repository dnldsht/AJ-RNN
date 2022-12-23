name                                   runs    accuracy    val_accuracy    test_accuracy    best_val_epoch    G_epoch    hidden_size    batch_size    learning_rate    dropout    train_time
-----------------------------------  ------  ----------  --------------  ---------------  ----------------  ---------  -------------  ------------  ---------------  ---------  ------------
G-LIGHT-GRU128-G1-LR3-B32-REG-DROP        1        98.6            22.1             30.9             169            1            128            32           1e-03         0.5           3.6
G-LIGHT-GRU128-G1-LR5-B32-REG             2        85.8            40.2             44.2              10.5          1            128            32           1e-05         0             2.9
G-LIGHT-GRU128-G1-LR6-B32-REG             2        86.1            49               54.3              89.5          1            128            32           1e-06         0             5.1
G-LIGHT-GRU128-G1-LR7-B32-REG             2        84.1            51.8             53               499            1            128            32           1e-07         0            14
G-LIGHT-GRU128-G1-LR8-B32-REG             3        85.5            56.8             57.1             665.3          1            128            32           1e-08         0            27.2
G-LIGHT-GRU128-G1-LR9-B32-REG-DROP        2        88.2            49               52.3            1000            1            128            32           1e-09         0.5          26.4
G-LIGHT-GRU128-G1-LR9-B32-REG             2        83.5            50.5             53.9             499.5          1            128            32           1e-09         0            17.8


G-LIGHT-GRU128-G1-LR3-B256-REG-DROP       1        78.6            77.7             78.5              26            1            128           256           1e-03         0.5           1
G-LIGHT-GRU128-G1-LR3-B256-REG            3        80.6            78.2             75.1             150.7          1            128           256           1e-03         0             7.1
G-LIGHT-GRU128-G1-LR4-B256-REG            2        91.5            87               86.6             353            1            128           256           1e-04         0             7.6
G-LIGHT-GRU128-G1-LR5-B256-REG            2        85.4            84               83.3             272.5          1            128           256           1e-05         0             5.7
G-LIGHT-GRU128-G1-LR6-B256-REG            2        84.7            82.7             85.6             744.5          1            128           256           1e-06         0             7.5
G-LIGHT-GRU128-G1-LR7-B256-REG-DROP       2        70.7            77.3             84               501            1            128           256           1e-07         0.5           7.9
G-LIGHT-GRU128-G1-LR7-B256-REG            2        82.6            81.2             81.8             998            1            128           256           1e-07         0            14.7