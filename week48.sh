python train.py --epoch 600 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256 --seed 196 --reg_classifier -results results/week48/GRU128-G1-LR3-BS256-REG/seed196

python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-6 --batch_size 256 --seed 89 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G5-LR6-BS256-REG-DROP/seed89
python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-6 --batch_size 256 --seed 196 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G5-LR6-BS256-REG-DROP/seed196


python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-6 --batch_size 32 --seed 89 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G5-LR6-BS32-REG-DROP/seed89
python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-6 --batch_size 32 --seed 196 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G5-LR6-BS32-REG-DROP/seed196


python train.py --epoch 600 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 32 --seed 196 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G1-LR3-BS32-REG-DROP/seed196

python train.py --epoch 600 --lamda_D 1 --G_epoch 1 --cell_type GRU --hidden_size 128 --learning_rate 1e-3 --batch_size 256 --seed 27836 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G1-LR3-BS256-REG-DROP/seed27836


python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --seed 89 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G1-LR4-BS32-REG-DROP/seed89
python train.py --epoch 600 --lamda_D 1 --G_epoch 5 --cell_type GRU --hidden_size 128 --learning_rate 1e-4 --batch_size 32 --seed 196 --reg_classifier --dropout 0.5 -results results/week48/GRU128-G1-LR4-BS32-REG-DROP/seed196