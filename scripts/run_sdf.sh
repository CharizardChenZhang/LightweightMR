GPU_IDX=0
SUBDARADIR=SDF
SDF_CHECKPOINT=ckpt_020000.pth

DATA_DIR=./example/data/
EXP_DIR=./example/exp/
for SCAN in "47984" "44234" "354371"
do
    CONF=./confs/sdf.conf
    python run_sdf.py --conf $CONF --mode train --subdatadir $SUBDARADIR --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --gpu $GPU_IDX
    python run_sdf.py --conf $CONF --mode validate_mesh --subdatadir $SUBDARADIR --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --gpu $GPU_IDX --checkpoint_name $SDF_CHECKPOINT
done