NUM_HIDDEN_LAYERS=${1:-4}
if [ $NUM_HIDDEN_LAYERS -gt 10 ]; then
    MEMORY=30G
else
    MEMORY=16G
fi

PRETRAIN_DIST="1,1,1,1,1"
SFT_DIST="1,1,0,0,0"
PERCENTAGES=("1perc" "5perc" "10perc" "20perc" "50perc" "100perc" "infperc")
NUM_TRAIN_EXAMPLES=1000
LEARNING_RATE=8e-5
NUM_TRAIN_EPOCHS=5

# train with different amounts of SFT examples
for PERCENTAGE in "${PERCENTAGES[@]}"
do
    case $PERCENTAGE in
        "infperc")
            NUM_SFT_EXAMPLES=0
            PRETRAIN_DIST=$SFT_DIST
            ;;
        *)
            NUM_SFT_EXAMPLES=$((NUM_TRAIN_EXAMPLES * ${PERCENTAGE%perc} / 20))
            PRETRAIN_DIST="1,1,1,1,1"
            ;;
    esac

    nlprun -n $PERCENTAGE -g 1 "python train.py --num_hidden_layers $NUM_HIDDEN_LAYERS \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --output_dir $PERCENTAGE-$NUM_HIDDEN_LAYERS \
        --num_train_examples $NUM_TRAIN_EXAMPLES \
        --num_sft_examples $NUM_SFT_EXAMPLES \
        --learning_rate $LEARNING_RATE \
        --pretrain_dist $PRETRAIN_DIST \
        --sft_dist $SFT_DIST" -r $MEMORY
done