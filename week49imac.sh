python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256  --seed 89 --reg_loss --dropout 0.5 --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR3-B256-REG-DROP/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256  --seed 89 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR3-B256-REG/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 32  --seed 89 --reg_loss --dropout 0.5 --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR3-B32-REG-DROP/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-8 --batch_size 32  --seed 89 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR8-B32-REG/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-8 --batch_size 32  --seed 196 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR8-B32-REG/seed196

