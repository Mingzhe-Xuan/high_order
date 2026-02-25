#!/bin/bash

# High-order Equivariant Network Training Script
# This script provides the same functionality as main.py by calling it with appropriate arguments

# Default values matching those in main.py
CUTOFF=5.0
BATCH_SIZE=32
PIN_MEMORY=true
NUM_WORKERS=0
SEED=42
TRAIN_VAL_TEST="0.8 0.1 0.1"

# Model parameters - embedding layer
DIST_EMB_FUNC="gaussian"
EMBED_DIM=64
MAX_ATOM_TYPE=118

# Model parameters - invariant layers
INV_UPDATE_METHOD="comformer"
NUM_INV_LAYERS=3

# Model parameters - middle MLP
MIDDLE_SCALAR_HIDDEN_DIM=128
NUM_MIDDLE_HIDDEN_LAYERS=1

# Model parameters - equivariant layers
EQUI_UPDATE_METHOD="tpconv_with_edge"
NUM_EQUI_LAYERS=3
TP_METHOD="so2"
SCALAR_DIM=16
VEC_DIM=8

# Model parameters - final MLP
NUM_FINAL_HIDDEN_LAYERS=1
FINAL_SCALAR_HIDDEN_DIM=64
FINAL_VEC_HIDDEN_DIM=16
FINAL_SCALAR_OUT_DIM=16
FINAL_VEC_OUT_DIM=8

# Training parameters
NEED_SELF_TRAIN=true
NEED_SCALAR_TRAIN=true
NEED_TENSOR_TRAIN=true
FINAL_POOLING=false
NUM_EPOCHS=100
LR=0.001
WEIGHT_DECAY=1e-05
CLIP_GRAD_NORM=1.0
SAVE_INTERVAL=5
OPTIMIZER="adamw"
SCHEDULER="cosine_annealing"
SELF_LOSS_FUNC="huber"
SCALAR_LOSS_FUNC="huber"
TENSOR_LOSS_FUNC="huber"
SELF_TRAIN_LIMIT=""
SCALAR_TRAIN_LIMIT=""
TENSOR_TRAIN_LIMIT=""

# Directory parameters
CHECKPOINT_DIR="checkpoints"
PIC_DIR="pics"
METRIC_DIR="metrics"
START_EPOCH=0

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -c, --cutoff FLOAT              Cutoff distance for graph construction (default: 5.0)"
    echo "  -b, --batch-size INT            Batch size for dataloaders (default: 32)"
    echo "  --pin-memory                    Enable pin memory for dataloaders (default: true)"
    echo "  --no-pin-memory                 Disable pin memory for dataloaders"
    echo "  -w, --num-workers INT           Number of workers for dataloaders (default: 0)"
    echo "  -s, --seed INT                  Random seed (default: 42)"
    echo "  --train-val-test F1 F2 F3       Train/validation/test split ratios (default: 0.8 0.1 0.1)"
    echo ""
    echo "Model Options:"
    echo "  --dist-emb-func STR             Distance embedding function (default: gaussian)"
    echo "  --embed-dim INT                 Embedding dimension (default: 64)"
    echo "  --max-atom-type INT             Maximum atom type (default: 118)"
    echo "  --inv-update-method STR         Invariant update method (default: conformer)"
    echo "  --num-inv-layers INT            Number of invariant layers (default: 3)"
    echo "  --middle-scalar-hidden-dim INT  Hidden dimension for middle scalar layers (default: 128)"
    echo "  --num-middle-hidden-layers INT  Number of middle hidden layers (default: 1)"
    echo "  --equi-update-method STR        Equivariant update method (default: tpconv_with_edge)"
    echo "  --num-equi-layers INT           Number of equivariant layers (default: 3)"
    echo "  --tp-method STR                 Tensor product method (default: so2)"
    echo "  --scalar-dim INT                Scalar dimension (default: 16)"
    echo "  --vec-dim INT                   Vector dimension (default: 8)"
    echo "  --num-final-hidden-layers INT   Number of final hidden layers (default: 1)"
    echo "  --final-scalar-hidden-dim INT   Final scalar hidden dimension (default: 64)"
    echo "  --final-vec-hidden-dim INT      Final vector hidden dimension (default: 16)"
    echo "  --final-scalar-out-dim INT      Final scalar output dimension (default: 16)"
    echo "  --final-vec-out-dim INT         Final vector output dimension (default: 8)"
    echo ""
    echo "Training Options:"
    echo "  --need-self-train               Enable self training (default: true)"
    echo "  --no-need-self-train            Disable self training"
    echo "  --need-scalar-train             Enable scalar training (default: true)"
    echo "  --no-need-scalar-train          Disable scalar training"
    echo "  --need-tensor-train             Enable tensor training (default: true)"
    echo "  --no-need-tensor-train          Disable tensor training"
    echo "  --final-pooling                 Enable final pooling (default: false)"
    echo "  --num-epochs INT                Number of training epochs (default: 100)"
    echo "  --lr FLOAT                      Learning rate (default: 0.001)"
    echo "  --weight-decay FLOAT            Weight decay (default: 1e-05)"
    echo "  --clip-grad-norm FLOAT          Gradient clipping norm (default: 1.0)"
    echo "  --save-interval INT             Model save interval (default: 5)"
    echo "  --optimizer STR                 Optimizer type (default: adamw)"
    echo "  --scheduler STR                 Learning rate scheduler (default: cosine_annealing)"
    echo "  --self-loss-func STR            Self loss function (default: huber)"
    echo "  --scalar-loss-func STR          Scalar loss function (default: huber)"
    echo "  --tensor-loss-func STR          Tensor loss function (default: huber)"
    echo "  --self-train-limit INT          Limit for self training samples (default: none)"
    echo "  --scalar-train-limit INT        Limit for scalar training samples (default: none)"
    echo "  --tensor-train-limit INT        Limit for tensor training samples (default: none)"
    echo ""
    echo "Directory Options:"
    echo "  --checkpoint-dir STR            Checkpoint directory (default: checkpoints)"
    echo "  --pic-dir STR                   Picture output directory (default: pics)"
    echo "  --metric-dir STR                Metrics output directory (default: metrics)"
    echo "  --start-epoch INT               Starting epoch for training (default: 0)"
    echo ""
    echo "Examples:"
    echo "  $0 --cutoff 6.0 --batch-size 64 --num-epochs 200"
    echo "  $0 --lr 0.0001 --optimizer adam --scheduler step"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cutoff)
            CUTOFF="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --pin-memory)
            PIN_MEMORY=true
            shift
            ;;
        --no-pin-memory)
            PIN_MEMORY=false
            shift
            ;;
        -w|--num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --train-val-test)
            TRAIN_VAL_TEST="$2 $3 $4"
            shift 4
            ;;
        --dist-emb-func)
            DIST_EMB_FUNC="$2"
            shift 2
            ;;
        --embed-dim)
            EMBED_DIM="$2"
            shift 2
            ;;
        --max-atom-type)
            MAX_ATOM_TYPE="$2"
            shift 2
            ;;
        --inv-update-method)
            INV_UPDATE_METHOD="$2"
            shift 2
            ;;
        --num-inv-layers)
            NUM_INV_LAYERS="$2"
            shift 2
            ;;
        --middle-scalar-hidden-dim)
            MIDDLE_SCALAR_HIDDEN_DIM="$2"
            shift 2
            ;;
        --num-middle-hidden-layers)
            NUM_MIDDLE_HIDDEN_LAYERS="$2"
            shift 2
            ;;
        --equi-update-method)
            EQUI_UPDATE_METHOD="$2"
            shift 2
            ;;
        --num-equi-layers)
            NUM_EQUI_LAYERS="$2"
            shift 2
            ;;
        --tp-method)
            TP_METHOD="$2"
            shift 2
            ;;
        --scalar-dim)
            SCALAR_DIM="$2"
            shift 2
            ;;
        --vec-dim)
            VEC_DIM="$2"
            shift 2
            ;;
        --num-final-hidden-layers)
            NUM_FINAL_HIDDEN_LAYERS="$2"
            shift 2
            ;;
        --final-scalar-hidden-dim)
            FINAL_SCALAR_HIDDEN_DIM="$2"
            shift 2
            ;;
        --final-vec-hidden-dim)
            FINAL_VEC_HIDDEN_DIM="$2"
            shift 2
            ;;
        --final-scalar-out-dim)
            FINAL_SCALAR_OUT_DIM="$2"
            shift 2
            ;;
        --final-vec-out-dim)
            FINAL_VEC_OUT_DIM="$2"
            shift 2
            ;;
        --need-self-train)
            NEED_SELF_TRAIN=true
            shift
            ;;
        --no-need-self-train)
            NEED_SELF_TRAIN=false
            shift
            ;;
        --need-scalar-train)
            NEED_SCALAR_TRAIN=true
            shift
            ;;
        --no-need-scalar-train)
            NEED_SCALAR_TRAIN=false
            shift
            ;;
        --need-tensor-train)
            NEED_TENSOR_TRAIN=true
            shift
            ;;
        --no-need-tensor-train)
            NEED_TENSOR_TRAIN=false
            shift
            ;;
        --final-pooling)
            FINAL_POOLING=true
            shift
            ;;
        --num-epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --clip-grad-norm)
            CLIP_GRAD_NORM="$2"
            shift 2
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --self-loss-func)
            SELF_LOSS_FUNC="$2"
            shift 2
            ;;
        --scalar-loss-func)
            SCALAR_LOSS_FUNC="$2"
            shift 2
            ;;
        --tensor-loss-func)
            TENSOR_LOSS_FUNC="$2"
            shift 2
            ;;
        --self-train-limit)
            SELF_TRAIN_LIMIT="--self-train-limit $2"
            shift 2
            ;;
        --scalar-train-limit)
            SCALAR_TRAIN_LIMIT="--scalar-train-limit $2"
            shift 2
            ;;
        --tensor-train-limit)
            TENSOR_TRAIN_LIMIT="--tensor-train-limit $2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --pic-dir)
            PIC_DIR="$2"
            shift 2
            ;;
        --metric-dir)
            METRIC_DIR="$2"
            shift 2
            ;;
        --start-epoch)
            START_EPOCH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Construct the Python command with all arguments
CMD="python src/main.py \
--cutoff $CUTOFF \
--batch-size $BATCH_SIZE \
--num-workers $NUM_WORKERS \
--seed $SEED \
--train-val-test $(echo $TRAIN_VAL_TEST) \
--dist-emb-func $DIST_EMB_FUNC \
--embed-dim $EMBED_DIM \
--max-atom-type $MAX_ATOM_TYPE \
--inv-update-method $INV_UPDATE_METHOD \
--num-inv-layers $NUM_INV_LAYERS \
--middle-scalar-hidden-dim $MIDDLE_SCALAR_HIDDEN_DIM \
--num-middle-hidden-layers $NUM_MIDDLE_HIDDEN_LAYERS \
--equi-update-method $EQUI_UPDATE_METHOD \
--num-equi-layers $NUM_EQUI_LAYERS \
--tp-method $TP_METHOD \
--scalar-dim $SCALAR_DIM \
--vec-dim $VEC_DIM \
--num-final-hidden-layers $NUM_FINAL_HIDDEN_LAYERS \
--final-scalar-hidden-dim $FINAL_SCALAR_HIDDEN_DIM \
--final-vec-hidden-dim $FINAL_VEC_HIDDEN_DIM \
--final-scalar-out-dim $FINAL_SCALAR_OUT_DIM \
--final-vec-out-dim $FINAL_VEC_OUT_DIM \
--num-epochs $NUM_EPOCHS \
--lr $LR \
--weight-decay $WEIGHT_DECAY \
--clip-grad-norm $CLIP_GRAD_NORM \
--save-interval $SAVE_INTERVAL \
--optimizer $OPTIMIZER \
--scheduler $SCHEDULER \
--self-loss-func $SELF_LOSS_FUNC \
--scalar-loss-func $SCALAR_LOSS_FUNC \
--tensor-loss-func $TENSOR_LOSS_FUNC \
--checkpoint-dir $CHECKPOINT_DIR \
--pic-dir $PIC_DIR \
--metric-dir $METRIC_DIR \
--start-epoch $START_EPOCH"

# Add boolean flags based on their values
if [ "$PIN_MEMORY" = true ]; then
    CMD="$CMD --pin-memory"
else
    CMD="$CMD --no-pin-memory"
fi

if [ "$NEED_SELF_TRAIN" = true ]; then
    CMD="$CMD --need-self-train"
else
    CMD="$CMD --no-need-self-train"
fi

if [ "$NEED_SCALAR_TRAIN" = true ]; then
    CMD="$CMD --need-scalar-train"
else
    CMD="$CMD --no-need-scalar-train"
fi

if [ "$NEED_TENSOR_TRAIN" = true ]; then
    CMD="$CMD --need-tensor-train"
else
    CMD="$CMD --no-need-tensor-train"
fi

if [ "$FINAL_POOLING" = true ]; then
    CMD="$CMD --final-pooling"
fi

# Add optional limits if they were specified
if [ ! -z "$SELF_TRAIN_LIMIT" ]; then
    CMD="$CMD $SELF_TRAIN_LIMIT"
fi

if [ ! -z "$SCALAR_TRAIN_LIMIT" ]; then
    CMD="$CMD $SCALAR_TRAIN_LIMIT"
fi

if [ ! -z "$TENSOR_TRAIN_LIMIT" ]; then
    CMD="$CMD $TENSOR_TRAIN_LIMIT"
fi

# Execute the command
echo "Executing: $CMD"
eval $CMD