
command_file=`basename "$0"`
gpu=0
model=model_cls_direct
data=processed_data
num_point=512
num_frame=4
batch_size=16
model_path=log_model_cls_direct_4_no_seagull_f4g0_new_splits_type_cls/model_epoch_85_mAcc_0.5618631716164924.ckpt
log_dir=log_${model}_${num_frame}_test_no_seagull_f4g0_new_splits_type_cls


python3 test.py \
    --gpu $gpu \
    --data $data \
    --model $model \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --batch_size $batch_size \
    --command_file $command_file \
    > $log_dir.txt 2>&1
