#python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256 --seed 89  -results results/week47/GRU128-LR4-BS256/seed89
#python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-5 --batch_size 256 --seed 89  -results results/week47/GRU128-LR5-BS256/seed89

python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 128 --seed 89  -results results/week47/GRU128-LR7-BS128/seed89
python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 64 --seed 89  -results results/week47/GRU128-LR7-BS64/seed89
python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 256 --seed 89  -results results/week47/GRU128-LR7-BS256/seed89
python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 32 --seed 89  -results results/week47/GRU128-LR7-BS32/seed89


