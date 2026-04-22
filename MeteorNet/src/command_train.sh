


command_file=`basename "$0"`
gpu=0
model=model_cls_direct
data=processed_data
num_point=512
num_frame=2
max_epoch=300
batch_size=16
learning_rate=0.001
# model_path=log_model_part_seg_step1/model-1.ckpt
model_path=None
log_dir=output/log_${model}_${num_frame}

# Ensure directory exists
mkdir -p "$(dirname "$log_dir")"

python3 action_cls/train.py \
    --gpu $gpu \
    --data $data \
    --model $model \
    --learning_rate $learning_rate \
    --model_path $model_path \
    --log_dir $log_dir \
    --num_point $num_point \
    --num_frame $num_frame \
    --max_epoch $max_epoch \
    --batch_size $batch_size \
    --command_file $command_file \
    > $log_dir.txt 2>&1
