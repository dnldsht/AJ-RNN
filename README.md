# Requirements
- python 3.9 (for tensorflow==2.6)

# README
It'a tensorflow imeplementaion of TPAMI2020 paper "Adversarial Joint-Learning Recurrent Neural Network for Incomplete Time Series Classification" . The paper can be found at https://ieeexplore.ieee.org/abstract/document/9210118.

Qianli Ma and Sen Li equally contributed to this work.

## To run your own model
```bash
python ajrnn.py --batch_size 20 --epoch 400 --lamda_D 1 --G_epoch 5 --train_data_filename xxx.csv --test_data_filename xxx.csv

# for example 
python ajrnn.py --batch_size 20 --epoch 400 --lamda_D 1 --G_epoch 5 --train_data_filename results/data/50words/50words_TRAIN_20.csv --test_data_filename results/data/50words/50words_TEST_20.csv

# to save you model add
--save path/to/model
```
## To load saved model
```bash
cd ./results
python ajrnn.py --dataset_name xxx --missing_ratio xxx

# for example 
python ajrnn.py --dataset_name Computers --missing_ratio 20
```


## Citation

If you find this repository, e.g., the code and the datasets, useful in your research, please cite the following paper:
```
@ARTICLE{9210118,  
author={Qianli {Ma} and Sen {Li} and Garrison W. {Cottrell}},  
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},   
title={Adversarial Joint-Learning Recurrent Neural Network for Incomplete Time Series Classification},   
year={2020},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/TPAMI.2020.3027975}}
```
$ python ajrnn.py --batch_size 20 --epoch 600 --lamda_D 1 --G_epoch 5 --train_data_filename results/data/Two_Patterns/Two_Patterns_TRAIN_20.csv --test_data_filename results/data/Two_Patterns/Two_Patterns_TEST_20.csv