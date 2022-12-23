python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256  --seed 23 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B256-REG/seed23


python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256 --dropout 0.5 --seed 196 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B256-REG-DROP5/seed196
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256 --dropout 0.5 --seed 89 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B256-REG-DROP5/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256 --dropout 0.8 --seed 196 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B256-REG-DROP8/seed196
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 256 --dropout 0.8 --seed 89 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B256-REG-DROP8/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --dropout 0 --seed 196 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B32-REG/seed196
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --dropout 0 --seed 89 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B32-REG/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --dropout 0 --seed 196 --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B32/seed196
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --dropout 0 --seed 89 --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B32/seed89

python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --dropout 0.5 --seed 196 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B32-REG-DROP5/seed196
python train.py --epoch 1000 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --dropout 0.5 --seed 89 --reg_loss --light_ajrnn -results results/week49/G-LIGHT-GRU128-G1-LR4-B32-REG-DROP5/seed89
