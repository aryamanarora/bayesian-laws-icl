$pretrain_dist="1,1,1,1,1"
$sft_dist="0,1,0,0,0"

nlprun -n 1perc-md -g 1 'python train.py --num_hidden_layers 8 --output_dir 1perc-8 --num_train_examples 1000 --num_sft_examples 50 --learning_rate 8e-5 --pretrain_dist $pretrain_dist --sft_dist $sft_dist'
nlprun -n 5perc-md -g 1 'python train.py --num_hidden_layers 8 --output_dir 5perc-8 --num_train_examples 1000 --num_sft_examples 250 --learning_rate 8e-5 --pretrain_dist $pretrain_dist --sft_dist $sft_dist'
nlprun -n 10perc-md -g 1 'python train.py --num_hidden_layers 8 --output_dir 10perc-8 --num_train_examples 1000 --num_sft_examples 500 --learning_rate 8e-5 --pretrain_dist $pretrain_dist --sft_dist $sft_dist'
nlprun -n 20perc-md -g 1 'python train.py --num_hidden_layers 8 --output_dir 20perc-8 --num_train_examples 1000 --num_sft_examples 1000 --learning_rate 8e-5 --pretrain_dist $pretrain_dist --sft_dist $sft_dist'
nlprun -n 50perc-md -g 1 'python train.py --num_hidden_layers 8 --output_dir 50perc-8 --num_train_examples 1000 --num_sft_examples 2500 --learning_rate 8e-5 --pretrain_dist $pretrain_dist --sft_dist $sft_dist'
nlprun -n 100perc-md -g 1 'python train.py --num_hidden_layers 8 --output_dir 100perc-8 --num_train_examples 1000 --num_sft_examples 5000 --learning_rate 8e-5 --pretrain_dist $pretrain_dist --sft_dist $sft_dist'
nlprun -n infperc-md -g 1 'python train.py --num_hidden_layers 8 --output_dir infperc-8 --num_train_examples 1000 --num_sft_examples 0 --pretrain_dist $sft_dist --learning_rate 8e-5'