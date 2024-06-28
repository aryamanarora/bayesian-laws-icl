NUM_HIDDEN_LAYERS=${1:-4}
if [ $NUM_HIDDEN_LAYERS -gt 10 ]; then
    MEMORY=30G
else
    MEMORY=16G
fi

PRETRAIN_DIST="1,1,1,1,1"
SFT_DIST="1,0,0,0,0"
PERCENTAGES=("1perc" "5perc" "10perc" "20perc" "50perc" "100perc")
NUM_TRAIN_EXAMPLES=1000
LEARNING_RATE=8e-5
NUM_TRAIN_EPOCHS=5

# create NUM_SFT_EXAMPLES
NUM_SFT_EXAMPLES=""
for PERCENTAGE in "${PERCENTAGES[@]}"
do
    PERCENTAGE_VALUE=${PERCENTAGE%perc}
    CALCULATED_VALUE=$((NUM_TRAIN_EXAMPLES * PERCENTAGE_VALUE * NUM_TRAIN_EPOCHS / 100))
    NUM_SFT_EXAMPLES+="$CALCULATED_VALUE,"
done

# remove trailing comma
NUM_SFT_EXAMPLES=${NUM_SFT_EXAMPLES%,}

# print num_sft_examples
echo $NUM_SFT_EXAMPLES

# train with different amounts of SFT examples
nlprun -n $NUM_HIDDEN_LAYERS-pretrain-sft -g 1 "python train.py --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $NUM_HIDDEN_LAYERS-$PRETRAIN_DIST-$SFT_DIST \
    --num_train_examples $NUM_TRAIN_EXAMPLES \
    --num_sft_examples $NUM_SFT_EXAMPLES \
    --learning_rate $LEARNING_RATE \
    --pretrain_dist $PRETRAIN_DIST \
    --sft_dist $SFT_DIST" -r $MEMORY

# train SFT dist separately
nlprun -n $NUM_HIDDEN_LAYERS-sft-only -g 1 "python train.py --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $NUM_HIDDEN_LAYERS-$SFT_DIST-$SFT_DIST \
    --num_train_examples $NUM_TRAIN_EXAMPLES \
    --num_sft_examples 0 \
    --learning_rate $LEARNING_RATE \
    --pretrain_dist $SFT_DIST \
    --sft_dist $SFT_DIST" -r $MEMORY