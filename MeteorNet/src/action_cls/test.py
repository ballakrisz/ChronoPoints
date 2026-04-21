'''
    Single-GPU training.
'''
import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1])) # Add the src directory to sys.path
from data.dataset import PointSeriesDataset


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

import msr_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_cls_ssg', help='Model name [default: pointnet2_cls_ssg]')
parser.add_argument('--model_path', default='', help='Model checkpint path [default: ]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--data', default='', help='Data path [default: ]')
parser.add_argument('--num_frames', type=int, default=1, help='Number of frames [default: 1]')
parser.add_argument('--skip_frames', type=int, default=1, help='Skip frames [default: 1]')
parser.add_argument('--max_epoch', type=int, default=251, help='Epoch to run [default: 251]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--command_file', default=None, help='Command file name [default: None]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_FRAME = FLAGS.num_frames
DATA = FLAGS.data
SKIP_FRAME = FLAGS.skip_frames
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
COMMAND_FILE = FLAGS.command_file

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(FLAGS.model+'.py')


BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

FRAME_GAP_DICT = {
    2 : 0,
    4 : 0,
    8 : 0,
    10 : 1,
    12 : 1,
    14 : 1,
    16 : 2,
    18 : 2,
    20 : 2,
}

config = "f{}g{}".format(FLAGS.num_frames, FRAME_GAP_DICT[FLAGS.num_frames])

TEST_DATASET = PointSeriesDataset(
    data_root_dir='/home/appuser/chrono_points_cls_benchmark',
    split='test',
    max_points_per_frame=512,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    sequence_format=config,
    backend="tf"
)

NUM_CLASSES = TEST_DATASET.num_types
print("Number of classes: ", NUM_CLASSES)


def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def test():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_FRAME)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, NUM_FRAME, is_training_pl,NUM_CLASSES, bn_decay=bn_decay)
            MODEL.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')
            total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('total_loss', total_loss)
            for l in losses:
                tf.summary.scalar(l.op.name, l)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # 🔥 Restore pretrained model if path provided
        if FLAGS.model_path and os.path.exists(FLAGS.model_path + ".index"):
            print("Restoring model from {}".format(FLAGS.model_path))
            saver.restore(sess, FLAGS.model_path)
        else:
            print("No checkpoint loaded — training from scratch")

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': total_loss,
               'accuracy': accuracy,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}
        

        val_oa, val_macc, val_class_acc = eval_one_epoch(sess, ops)
        print("Final Eval: OA={:.4f}, mAcc={:.4f}".format(
            val_oa, val_macc))
        print("Class-wise Accuracy:")
        for i, acc in enumerate(val_class_acc):
            print("Class {}: {:.4f}".format(i, acc))


def get_batch(dataset, start_idx, end_idx):
    bsize = end_idx-start_idx
    # assert(NUM_FRAME==1)
    batch_data = np.zeros((bsize, NUM_POINT * NUM_FRAME, 3))
    batch_label = np.zeros((bsize,), dtype=np.int32)
    batch_seqid = [0]*bsize
    for i in range(bsize):
        data, label, seq_id = dataset[i+start_idx]
        batch_data[i] = np.reshape(data, [-1, 3])
        batch_label[i] = label
        batch_seqid[i] = seq_id
    return batch_data, batch_label, batch_seqid


def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    # log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT*NUM_FRAME, 3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    num_batches = (len(TEST_DATASET)-1) // BATCH_SIZE + 1

    total_correct = 0
    total_seen = 0
    total_loss = 0.0

    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_seen = np.zeros(NUM_CLASSES)

    # For majority-vote sequence accuracy
    per_seq_vote = {}
    per_seq_label = {}

    for batch_idx in range(num_batches):
        batch_data, batch_label, batch_seqid = get_batch(
            TEST_DATASET,
            batch_idx * BATCH_SIZE,
            min((batch_idx+1) * BATCH_SIZE, len(TEST_DATASET))
        )

        bsize = batch_data.shape[0]
        cur_batch_data[:bsize] = batch_data
        cur_batch_label[:bsize] = batch_label

        feed_dict = {
            ops['pointclouds_pl']: cur_batch_data,
            ops['labels_pl']: cur_batch_label,
            ops['is_training_pl']: is_training
        }

        summary, step, loss_val, pred_val = sess.run(
            [ops['merged'], ops['step'], ops['loss'], ops['pred']],
            feed_dict=feed_dict
        )

        # ----- softmax for sequence voting -----
        def softmax(arr):
            arr = arr - np.max(arr)
            return np.exp(arr) / np.sum(np.exp(arr))

        for i in range(bsize):
            seq_id = batch_seqid[i]
            prob = softmax(pred_val[i])
            if seq_id not in per_seq_vote:
                per_seq_vote[seq_id] = prob
                per_seq_label[seq_id] = batch_label[i]
            else:
                per_seq_vote[seq_id] += prob

        # ======== Compute sample-level accuracy ========
        pred_classes = np.argmax(pred_val, axis=1)
        correct = np.sum(pred_classes[:bsize] == batch_label[:bsize])

        total_correct += correct
        total_seen += bsize
        total_loss += loss_val

        # Per-class tracking
        for i in range(bsize):
            lbl = batch_label[i]
            per_class_seen[lbl] += 1
            per_class_correct[lbl] += (pred_classes[i] == lbl)

    # =====================================================
    #      🔥 Compute Evaluation Metrics (new version)
    # =====================================================

    # ---------- Overall Accuracy ----------
    overall_accuracy = total_correct / float(total_seen)

    # ---------- Mean Class Accuracy ----------
    class_acc = np.divide(
        per_class_correct,
        per_class_seen,
        out=np.zeros_like(per_class_correct, dtype=float),
        where=(per_class_seen != 0)
    )
    mean_class_accuracy = np.mean(class_acc)

    # ---------- Per-sequence accuracy ----------
    seq_predictions = {k: np.argmax(v) for k, v in per_seq_vote.items()}
    seq_correct = [seq_predictions[k] == per_seq_label[k] for k in seq_predictions]
    seq_accuracy = np.mean(seq_correct)

    # Per-sequence per-class accuracy
    per_seq_seen = np.zeros(NUM_CLASSES)
    per_seq_correct_class = np.zeros(NUM_CLASSES)
    for sid, pred_cls in seq_predictions.items():
        lbl = per_seq_label[sid]
        per_seq_seen[lbl] += 1
        per_seq_correct_class[lbl] += (pred_cls == lbl)

    seq_class_acc = np.divide(
        per_seq_correct_class,
        per_seq_seen,
        out=np.zeros_like(per_seq_correct_class, dtype=float),
        where=(per_seq_seen != 0)
    )

    EPOCH_CNT += 1
    return overall_accuracy, mean_class_accuracy, class_acc




if __name__ == "__main__":
    print('pid: %s'%(str(os.getpid())))
    test()
