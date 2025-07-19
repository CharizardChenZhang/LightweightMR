GPU_IDX=1
SDF_SUBDATADIR=SDF
SDF_CHECKPOINT=ckpt_020000.pth
SUBDATADIR=VG
VG_CHECKPOINT=ckpt_008000.pth

DATA_DIR=./example/data/
EXP_DIR=./example/exp/
for SCAN in "47984" "44234" "354371"
do
    CONF="./confs/vg.conf"
    python run_vg.py --conf $CONF --mode train --sdf_subdatadir $SDF_SUBDATADIR --sdf_checkpoint_name $SDF_CHECKPOINT \
    --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --subdatadir $SUBDATADIR --gpu $GPU_IDX
    python run_vg.py --conf $CONF --mode validate_mesh_delaunay --sdf_subdatadir $SDF_SUBDATADIR --sdf_checkpoint_name $SDF_CHECKPOINT \
    --datadir $DATA_DIR --expdir $EXP_DIR --dataname $SCAN --subdatadir $SUBDATADIR --gpu $GPU_IDX --checkpoint_name $VG_CHECKPOINT
done
