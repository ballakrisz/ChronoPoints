
command_file=`basename "$0"`
gpu=0
model=model_cls_direct
data=processed_data
num_point=512
num_frame=2
batch_size=16
model_path=log_${model}_${num_frame}/best_model.ckpt
log_dir=log_${model}_${num_frame}_test


python3 action_cls/test.py \
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
