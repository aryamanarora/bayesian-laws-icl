NUM_HIDDEN_LAYERS=${1:-4}
if [ $NUM_HIDDEN_LAYERS -gt 10 ]; then
    MEMORY=30G
else
    MEMORY=16G
fi

TRAINING_OPTS=""
if [ $NUM_HIDDEN_LAYERS -gt 12 ]; then
    TRAINING_OPTS="--per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 2"
fi

PRETRAIN_DIST="1,1,1,1,1"
SFT_DIST=${2:-"1,0,0,0,0"}
SFT_METHOD=${3:-"sft"}
PERCENTAGES=("1perc" "5perc" "10perc" "20perc" "50perc" "100perc")
NUM_TRAIN_EXAMPLES=${4:-1000}
NUM_TRAIN_EPOCHS=${5:-5}
MACHINE=${6:-""}

if [ "$MACHINE" != "" ]; then
    MACHINE=" -m $MACHINE"
fi

# create NUM_SFT_EXAMPLES
NUM_SFT_EXAMPLES=""
NUM_DPO_EXAMPLES=""
for PERCENTAGE in "${PERCENTAGES[@]}"
do
    PERCENTAGE_VALUE=${PERCENTAGE%perc}
    CALCULATED_VALUE=$((NUM_TRAIN_EXAMPLES * PERCENTAGE_VALUE * NUM_TRAIN_EPOCHS / 100))
    NUM_SFT_EXAMPLES+="$CALCULATED_VALUE,"
    CALCULATED_DPO_VALUE=$((CALCULATED_VALUE)) # dpo docs are 1/10th the length of sft docs
    NUM_DPO_EXAMPLES+="$CALCULATED_DPO_VALUE,"
done

# remove trailing comma
NUM_SFT_EXAMPLES=${NUM_SFT_EXAMPLES%,}
NUM_DPO_EXAMPLES=${NUM_DPO_EXAMPLES%,}

# print num_sft_examples
echo $NUM_SFT_EXAMPLES

if [ $SFT_METHOD == "none" ]; then

    # train with different amounts of SFT examples
    nlprun -n $NUM_HIDDEN_LAYERS-pretrain-sft -g 1 "uv run bayesian_laws_icl/train.py --num_hidden_layers $NUM_HIDDEN_LAYERS \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --output_dir $NUM_HIDDEN_LAYERS-$PRETRAIN_DIST-$SFT_DIST \
        --num_train_examples $NUM_TRAIN_EXAMPLES \
        --num_sft_examples 0 \
        --learning_rate 8e-5 \
        --pretrain_dist $PRETRAIN_DIST \
        --sft_dist $SFT_DIST \
        --sft_method sft \
        $TRAINING_OPTS" -r $MEMORY$MACHINE

fi


if [ $SFT_METHOD == "sft" ] || [ $SFT_METHOD == "sft,dpo" ]; then

    # train with different amounts of SFT examples
    nlprun -n $NUM_HIDDEN_LAYERS-pretrain-sft -g 1 "uv run bayesian_laws_icl/train.py --num_hidden_layers $NUM_HIDDEN_LAYERS \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --output_dir $NUM_HIDDEN_LAYERS-$PRETRAIN_DIST-$SFT_DIST \
        --num_train_examples $NUM_TRAIN_EXAMPLES \
        --num_sft_examples $NUM_SFT_EXAMPLES \
        --learning_rate 8e-5 \
        --pretrain_dist $PRETRAIN_DIST \
        --sft_dist $SFT_DIST \
        --sft_method sft \
        $TRAINING_OPTS" -r $MEMORY$MACHINE

    # train SFT dist separately
    nlprun -n $NUM_HIDDEN_LAYERS-sft-only -g 1 "uv run bayesian_laws_icl/train.py --num_hidden_layers $NUM_HIDDEN_LAYERS \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --output_dir $NUM_HIDDEN_LAYERS-$SFT_DIST-$SFT_DIST \
        --num_train_examples $NUM_TRAIN_EXAMPLES \
        --num_sft_examples 0 \
        --learning_rate 8e-5 \
        --pretrain_dist $SFT_DIST \
        --sft_dist $SFT_DIST \
        --sft_method sft \
        $TRAINING_OPTS" -r $MEMORY$MACHINE

fi

if [ $SFT_METHOD == "dpo" ] || [ $SFT_METHOD == "sft,dpo" ]; then

    nlprun -n $NUM_HIDDEN_LAYERS-pretrain-dpo -g 1 "uv run bayesian_laws_icl/train.py --num_hidden_layers $NUM_HIDDEN_LAYERS \
        --num_train_epochs $NUM_TRAIN_EPOCHS \
        --output_dir $NUM_HIDDEN_LAYERS-$PRETRAIN_DIST-$SFT_DIST-dpo \
        --load_dir logs/$NUM_HIDDEN_LAYERS-$PRETRAIN_DIST-$SFT_DIST \
        --num_train_examples $NUM_TRAIN_EXAMPLES \
        --num_sft_examples $NUM_DPO_EXAMPLES \
        --learning_rate 8e-6 \
        --pretrain_dist $PRETRAIN_DIST \
        --sft_dist $SFT_DIST \
        --sft_method $SFT_METHOD \
        --do_pretrain False \
        $TRAINING_OPTS" -r $MEMORY$MACHINE

fi
