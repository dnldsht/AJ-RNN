python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-9 --batch_size 32  --seed 89 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR9-B32-REG/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 256  --seed 89 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR7-B256-REG/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-9 --batch_size 32  --seed 196 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR9-B32-REG/seed196
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 256  --seed 196 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR7-B256-REG/see196

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-9 --batch_size 32 --dropout 0.5 --seed 89 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR9-B32-REG-DROP/seed89
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 256 --dropout 0.5 --seed 89 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR7-B256-REG-DROP/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-9 --batch_size 32 --dropout 0.5 --seed 196 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR9-B32-REG-DROP/seed196
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-7 --batch_size 256 --dropout 0.5 --seed 196 --reg_loss --light_ajrnn -results results/week50/G-LIGHT-GRU128-G1-LR7-B256-REG-DROP/see196

