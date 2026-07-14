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
import time

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

def compute_model_complexity_tf(
    sess,
    ops,
    batch_data,
    batch_label
):
    """
    TensorFlow 1.x model complexity + inference benchmark
    """

    # ====================================================
    # Parameters
    # ====================================================

    total_params = np.sum([
        np.prod(v.shape.as_list())
        for v in tf.trainable_variables()
    ])

    print("-------------------- SANITY CHECK --------------------")
    print("Total trainable parameters: {:,}".format(total_params))
    print("------------------------------------------------------")

    # ====================================================
    # FLOPs
    # ====================================================

    try:

        from tensorflow.python.profiler import model_analyzer
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder

        flops = tf.profiler.profile(
            tf.get_default_graph(),
            options=tf.profiler.ProfileOptionBuilder.float_operation()
        )

        total_flops = flops.total_float_ops

        print("\n---------------- MODEL COMPLEXITY ----------------")
        print("FLOPs:  {:.4f} GFLOPs".format(total_flops / 1e9))

        # MACs ≈ FLOPs / 2
        print("MACs:   {:.4f} GMACs".format(total_flops / 2e9))

        print("Params: {:.4f} M".format(total_params / 1e6))
        print("--------------------------------------------------")

    except Exception as e:

        print("FLOP computation failed:")
        print(e)

    # ====================================================
    # Benchmark
    # ====================================================

    warmup_iters = 20
    benchmark_iters = 100

    print("\nWarming up GPU...")

    feed_dict = {
        ops['pointclouds_pl']: batch_data,
        ops['labels_pl']: batch_label,
        ops['is_training_pl']: False
    }

    # --------------------------------
    # Warmup
    # --------------------------------

    for _ in range(warmup_iters):

        sess.run(
            ops['pred'],
            feed_dict=feed_dict
        )

    # force sync
    sess.run(tf.no_op())

    # --------------------------------
    # Benchmark
    # --------------------------------

    timings = []

    for _ in range(benchmark_iters):

        start = time.perf_counter()

        sess.run(
            ops['pred'],
            feed_dict=feed_dict
        )

        # force GPU sync
        sess.run(tf.no_op())

        end = time.perf_counter()

        timings.append(end - start)

    timings = np.array(timings)

    # ====================================================
    # Statistics
    # ====================================================

    mean_ms = timings.mean() * 1000
    std_ms = timings.std() * 1000
    median_ms = np.median(timings) * 1000
    min_ms = timings.min() * 1000
    max_ms = timings.max() * 1000

    fps = 1000.0 / mean_ms

    print("\n---------------- INFERENCE SPEED ----------------")
    print("Input shape:      {}".format(batch_data.shape))
    print("Warmup iterations:{}".format(warmup_iters))
    print("Benchmark runs:   {}".format(benchmark_iters))
    print("")
    print("Mean latency:     {:.3f} ms".format(mean_ms))
    print("Median latency:   {:.3f} ms".format(median_ms))
    print("Std latency:      {:.3f} ms".format(std_ms))
    print("Min latency:      {:.3f} ms".format(min_ms))
    print("Max latency:      {:.3f} ms".format(max_ms))
    print("Throughput:       {:.2f} FPS".format(fps))
    print("--------------------------------------------------\n")

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
    print("")
    print("!!! =================================================== IMPORTANT ==================================================== !!!")
    print("!!!     Because TensorFlow 1.x uses static graph construction, benchmark latency depends on the global BATCH_SIZE.     !!!")
    print("!!!    Unlike PyTorch, we cannot simply extract a single sample from a larger batch and benchmark only that sample.    !!!")
    print("!!! For fair single-sample benchmarking, BATCH_SIZE must be set to 1 in src/command_test.sh before building the graph. !!!")
    print("!!! =================================================== IMPORTANT ==================================================== !!!")
    print("")
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
        
        # ====================================================
        # Complexity benchmark
        # ====================================================

        sample_data, sample_label, _ = get_batch(
            TEST_DATASET,
            0,
            BATCH_SIZE
        )

        compute_model_complexity_tf(
            sess,
            ops,
            sample_data,
            sample_label
        )

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

    cur_batch_data = np.zeros((BATCH_SIZE, NUM_POINT*NUM_FRAME, 3))
    cur_batch_label = np.zeros((BATCH_SIZE), dtype=np.int32)
    num_batches = (len(TEST_DATASET)-1) // BATCH_SIZE + 1

    total_correct = 0
    total_seen = 0
    total_loss = 0.0

    per_class_correct = np.zeros(NUM_CLASSES)
    per_class_seen = np.zeros(NUM_CLASSES)

    # 🔥 NEW: confusion matrix
    confusion_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

    # =========================
    # Inference timing
    # =========================
    total_inference_time = 0.0

    # Sequence voting
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

        # =========================
        # Measure inference time
        # =========================
        start_time = time.perf_counter()

        summary, step, loss_val, pred_val = sess.run(
            [ops['merged'], ops['step'], ops['loss'], ops['pred']],
            feed_dict=feed_dict
        )

        end_time = time.perf_counter()

        batch_inference_time = end_time - start_time
        total_inference_time += batch_inference_time

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

        # ===== Sample-level =====
        pred_classes = np.argmax(pred_val, axis=1)

        for i in range(bsize):
            true = batch_label[i]
            pred = pred_classes[i]

            total_seen += 1
            total_correct += (pred == true)

            per_class_seen[true] += 1
            per_class_correct[true] += (pred == true)

            # 🔥 NEW: confusion matrix update
            confusion_counts[true, pred] += 1

        total_loss += loss_val

    # =========================
    # Accuracy
    # =========================
    overall_accuracy = total_correct / float(total_seen)

    class_acc = np.divide(
        per_class_correct,
        per_class_seen,
        out=np.zeros_like(per_class_correct, dtype=float),
        where=(per_class_seen != 0)
    )
    mean_class_accuracy = np.mean(class_acc)

    # =========================
    # Precision / Recall / F1
    # =========================
    tp = np.diag(confusion_counts)
    fp = np.sum(confusion_counts, axis=0) - tp
    fn = np.sum(confusion_counts, axis=1) - tp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-8)

    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)

    tp_sum = np.sum(tp)
    fp_sum = np.sum(fp)
    fn_sum = np.sum(fn)

    micro_precision = tp_sum / max(tp_sum + fp_sum, 1)
    micro_recall = tp_sum / max(tp_sum + fn_sum, 1)
    micro_f1 = 2 * micro_precision * micro_recall / max(micro_precision + micro_recall, 1e-8)

    # =========================
    # Sequence metrics (unchanged)
    # =========================
    seq_predictions = {}
    for k in per_seq_vote:
        seq_predictions[k] = np.argmax(per_seq_vote[k])

    seq_correct = []
    for k in seq_predictions:
        seq_correct.append(seq_predictions[k] == per_seq_label[k])

    seq_accuracy = np.mean(seq_correct)

    per_seq_seen = np.zeros(NUM_CLASSES)
    per_seq_correct_class = np.zeros(NUM_CLASSES)

    for sid in seq_predictions:
        lbl = per_seq_label[sid]
        pred_cls = seq_predictions[sid]

        per_seq_seen[lbl] += 1
        per_seq_correct_class[lbl] += (pred_cls == lbl)

    seq_class_acc = np.divide(
        per_seq_correct_class,
        per_seq_seen,
        out=np.zeros_like(per_seq_correct_class, dtype=float),
        where=(per_seq_seen != 0)
    )

    # =========================
    # Logging (print)
    # =========================
    print("\n=== Evaluation Summary ===")
    print("Overall Accuracy: {:.4f}".format(overall_accuracy))
    print("Mean Accuracy:    {:.4f}".format(mean_class_accuracy))

    print("\n--- Per-Class Accuracy ---")
    for c in range(NUM_CLASSES):
        print("Class {:2d} | Acc: {:.4f} | Seen: {:5d}".format(
            c, class_acc[c], int(per_class_seen[c])))

    print("\n--- Per-Class Precision / Recall / F1 ---")
    for c in range(NUM_CLASSES):
        print("Class {:2d} | P: {:.3f} | R: {:.3f} | F1: {:.3f}".format(
            c, precision[c], recall[c], f1[c]))

    print("\n--- Macro ---")
    print("Precision: {:.4f}".format(macro_precision))
    print("Recall:    {:.4f}".format(macro_recall))
    print("F1:        {:.4f}".format(macro_f1))

    print("\n--- Micro ---")
    print("Precision: {:.4f}".format(micro_precision))
    print("Recall:    {:.4f}".format(micro_recall))
    print("F1:        {:.4f}".format(micro_f1))

    print("\n--- Confusion Matrix ---")
    print(confusion_counts)

    print("\n--- Prediction Distribution per True Class ---")
    for c_true in range(NUM_CLASSES):
        print("Class {}:".format(c_true))
        for c_pred in range(NUM_CLASSES):
            count = confusion_counts[c_true, c_pred]
            if count > 0:
                print("   -> Pred {}: {}".format(c_pred, count))

    print("\n--- Sequence Accuracy ---")
    print("Seq OA: {:.4f}".format(seq_accuracy))

    print("==========================================\n")

    EPOCH_CNT += 1

    return overall_accuracy, mean_class_accuracy, class_acc




if __name__ == "__main__":
    print('pid: %s'%(str(os.getpid())))
    test()
