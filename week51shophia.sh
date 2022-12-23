python train.py --epoch 1000 --lamda_D 1 --G_epoch 2 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256 --dropout 0.5 --seed 89 --reg_loss -results results/week49/GRU128-G2-LR4-BS256-REG-DROP/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 2 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256 --dropout 0.5 --seed 196 --reg_loss -results results/week49/GRU128-G2-LR4-BS256-REG-DROP/seed196

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256 --dropout 0 --seed 89 --reg_loss -results results/week49/GRU128-G1-LR3-BS256-REG/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256 --dropout 0 --seed 196 --reg_loss -results results/week49/GRU128-G1-LR3-BS256-REG/seed196

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256 --dropout 0.8 --seed 89 --reg_loss -results results/week49/GRU128-G1-LR3-BS256-REG-DROP8/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256 --dropout 0.8 --seed 196 --reg_loss -results results/week49/GRU128-G1-LR3-BS256-REG-DROP8/seed196

#plan

## 


##
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 32 --seed 89 -results results/week49/GRU128-G1-LR7-BS32/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 2 --cell_type GRU --hidden_size 128 --learning_rate 1e-8 --batch_size 32 --seed 89 -results results/week49/GRU128-G2-LR8-BS32/seed89

