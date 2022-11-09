# GRU-100-L7
python train.py --batch_size 256 --epoch 600 --lamda_D 1 --G_epoch 5 --seed 89 --learning_rate 1e-7 -results results/gru100-seed89-lr7

# GRU-256-L8
python train.py --batch_size 256 --epoch 600 --lamda_D 1 --G_epoch 5 --seed 89 --learning_rate 1e-8 --hidden_size 256 -results results/gru256-seed89-lr8

# LSTM-257-L8
python train.py --batch_size 256 --epoch 600 --lamda_D 1 --G_epoch 5 --seed 89 --learning_rate 1e-8 --hidden_size 256 --cell_type LSTM -results results/lstm256-seed89-lr8