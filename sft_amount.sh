nlprun -n 1perc -g 1 'python train.py --output_dir 1perc --num_train_examples 1000 --num_sft_examples 50 --learning_rate 8e-5'
nlprun -n 5perc -g 1 'python train.py --output_dir 5perc --num_train_examples 1000 --num_sft_examples 250 --learning_rate 8e-5'
nlprun -n 10perc -g 1 'python train.py --output_dir 10perc --num_train_examples 1000 --num_sft_examples 500 --learning_rate 8e-5'
nlprun -n 20perc -g 1 'python train.py --output_dir 20perc --num_train_examples 1000 --num_sft_examples 1000 --learning_rate 8e-5'
nlprun -n 50perc -g 1 'python train.py --output_dir 50perc --num_train_examples 1000 --num_sft_examples 2500 --learning_rate 8e-5'
nlprun -n 100perc -g 1 'python train.py --output_dir 100perc --num_train_examples 1000 --num_sft_examples 5000 --learning_rate 8e-5'
nlprun -n infperc -g 1 'python train.py --output_dir infperc --num_train_examples 1000 --num_sft_examples 0 --pretrain_dist 1,0,0,0,0 --learning_rate 8e-5'