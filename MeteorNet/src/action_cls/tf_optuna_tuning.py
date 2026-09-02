# tf_optuna_tuning.py

import os
import sys
import random
import argparse
import numpy as np
import tensorflow as tf
import optuna
import importlib
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from optuna.exceptions import TrialPruned

# Add paths to your project (adjust as needed)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, '..', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'datasets'))

from data.dataset import PointSeriesDataset

# ============================================================
# Parse arguments (similar to your original train script)
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument('--model', default='model_cls_direct', help='Model name')
parser.add_argument('--num_point', type=int, default=512, help='Number of points per frame')
parser.add_argument('--num_frames', type=int, default=12, help='Number of frames')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--max_epoch', type=int, default=100, help='Epochs per trial')
parser.add_argument('--data_root', default='/home/appuser/LIFT_benchmark', help='Dataset root')
parser.add_argument('--target_trials', type=int, default=50, help='Total number of trials')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
FLAGS = parser.parse_args()

# ============================================================
# Fixed configuration
# ============================================================
os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
SEED = FLAGS.seed
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)

NUM_FRAME = FLAGS.num_frames
NUM_POINT = FLAGS.num_point
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
DATA_ROOT = FLAGS.data_root
TARGET_TRIALS = FLAGS.target_trials
MODEL_NAME = FLAGS.model

# Frame gap mapping (same as your original)
FRAME_GAP_DICT = {
    2: 0, 4: 0, 8: 0, 10: 1, 12: 1,
    14: 1, 16: 2, 18: 2, 20: 2,
}
config = "f{}g{}".format(NUM_FRAME, FRAME_GAP_DICT.get(NUM_FRAME, 0))

# ============================================================
# Load datasets (preloaded, reused across trials)
# ============================================================
print("Loading datasets...")
TRAIN_DATASET = PointSeriesDataset(
    data_root_dir=DATA_ROOT,
    split='train',
    max_points_per_frame=NUM_POINT,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    sequence_format=config,
    backend="tf"
)

TEST_DATASET = PointSeriesDataset(
    data_root_dir=DATA_ROOT,
    split='val',
    max_points_per_frame=NUM_POINT,
    single_return_only=True,
    preload=True,
    sampling_strategy="farthest",
    padding_strategy="zero_padding",
    sequence_format=config,
    backend="tf"
)

NUM_CLASSES = TRAIN_DATASET.num_types
print("Number of classes: {}".format(NUM_CLASSES))

# ============================================================
# Output directories
# ============================================================
OUTPUT_DIR = Path("./optuna_results_tf_{}".format(config))
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = OUTPUT_DIR / "best_models"
MODEL_DIR.mkdir(exist_ok=True)
DB_PATH = OUTPUT_DIR / "optuna.db"

# ============================================================
# Helper functions (from your original training script)
# ============================================================
def get_batch(dataset, start_idx, end_idx, num_point=NUM_POINT, num_frame=NUM_FRAME):
    bsize = end_idx - start_idx
    batch_data = np.zeros((bsize, num_point * num_frame, 3), dtype=np.float32)
    batch_label = np.zeros((bsize,), dtype=np.int32)
    batch_seqid = [0] * bsize
    for i in range(bsize):
        data, label, seq_id = dataset[start_idx + i]
        batch_data[i] = np.reshape(data, [-1, 3])
        batch_label[i] = label
        batch_seqid[i] = seq_id
    return batch_data, batch_label, batch_seqid

def train_one_epoch(sess, ops, train_writer, epoch, train_dataset, num_classes, batch_size):
    is_training = True
    losses = []
    mean_batch_acc = []
    total_correct = 0
    total_seen = 0
    per_class_correct = np.zeros(num_classes)
    per_class_seen = np.zeros(num_classes)

    num_batch = len(train_dataset) // batch_size
    for batch_idx in range(num_batch):
        batch_data, batch_label, _ = get_batch(
            train_dataset,
            batch_idx * batch_size,
            (batch_idx + 1) * batch_size
        )
        feed_dict = {
            ops['pointclouds_pl']: batch_data,
            ops['labels_pl']: batch_label,
            ops['is_training_pl']: is_training,
        }
        summary, step, _, loss_val, pred_val, acc_val = sess.run(
            [ops['merged'], ops['step'], ops['train_op'], ops['loss'],
             ops['pred'], ops['accuracy']],
            feed_dict=feed_dict
        )
        train_writer.add_summary(summary, step)
        losses.append(loss_val)
        mean_batch_acc.append(acc_val)

        pred_classes = np.argmax(pred_val, axis=1)
        correct = np.sum(pred_classes == batch_label)
        total_correct += correct
        total_seen += batch_label.shape[0]
        for i in range(batch_label.shape[0]):
            lbl = batch_label[i]
            per_class_seen[lbl] += 1
            per_class_correct[lbl] += (pred_classes[i] == lbl)

    mean_loss = np.mean(losses)
    overall_accuracy = total_correct / float(total_seen)
    class_acc = np.divide(
        per_class_correct,
        per_class_seen,
        out=np.zeros_like(per_class_correct, dtype=float),
        where=(per_class_seen != 0)
    )
    mean_class_acc = np.mean(class_acc)
    train_dataset.shuffle()
    return mean_loss, overall_accuracy, mean_class_acc

def eval_one_epoch(sess, ops, test_writer, epoch, test_dataset, num_classes, batch_size):
    is_training = False
    num_batches = (len(test_dataset) - 1) // batch_size + 1
    total_correct = 0
    total_seen = 0
    total_loss = 0.0
    per_class_correct = np.zeros(num_classes)
    per_class_seen = np.zeros(num_classes)

    per_seq_vote = {}
    per_seq_label = {}

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min((batch_idx + 1) * batch_size, len(test_dataset))
        batch_data, batch_label, batch_seqid = get_batch(test_dataset, start, end)
        bsize = batch_data.shape[0]

        cur_batch_data = np.zeros((batch_size, batch_data.shape[1], 3), dtype=np.float32)
        cur_batch_label = np.zeros((batch_size,), dtype=np.int32)
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
        test_writer.add_summary(summary, step)

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

        pred_classes = np.argmax(pred_val[:bsize], axis=1)
        correct = np.sum(pred_classes == batch_label)
        total_correct += correct
        total_seen += bsize
        total_loss += loss_val
        for i in range(bsize):
            lbl = batch_label[i]
            per_class_seen[lbl] += 1
            per_class_correct[lbl] += (pred_classes[i] == lbl)

    overall_accuracy = total_correct / float(total_seen)
    class_acc = np.divide(
        per_class_correct,
        per_class_seen,
        out=np.zeros_like(per_class_correct, dtype=float),
        where=(per_class_seen != 0)
    )
    mean_class_acc = np.mean(class_acc)
    return mean_class_acc

# ============================================================
# Optuna objective
# ============================================================
def objective(trial):
    # ---- Suggest hyperparameters ----
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-1)
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
    momentum = None
    if optimizer_name == "SGD":
        momentum = trial.suggest_uniform("momentum", 0.5, 0.99)

    # ---- Build fresh graph and session ----
    tf.reset_default_graph()
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(FLAGS.gpu)):
            # Import the model module
            model_module = importlib.import_module(MODEL_NAME)
            # Placeholders
            pointclouds_pl, labels_pl = model_module.placeholder_inputs(
                BATCH_SIZE, NUM_POINT, NUM_FRAME
            )
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Batch counter and BN decay (as in original)
            batch = tf.get_variable('batch', [],
                                    initializer=tf.constant_initializer(0),
                                    trainable=False)
            BN_INIT_DECAY = 0.5
            BN_DECAY_DECAY_RATE = 0.5
            BN_DECAY_DECAY_STEP = 200000.0
            BN_DECAY_CLIP = 0.99
            bn_momentum = tf.train.exponential_decay(
                BN_INIT_DECAY,
                batch * BATCH_SIZE,
                BN_DECAY_DECAY_STEP,
                BN_DECAY_DECAY_RATE,
                staircase=True
            )
            bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)

            # Forward pass
            pred, end_points = model_module.get_model(
                pointclouds_pl,
                NUM_FRAME,
                is_training_pl,
                NUM_CLASSES,
                bn_decay=bn_decay
            )

            # Loss from model (cross-entropy)
            model_module.get_loss(pred, labels_pl, end_points)
            losses = tf.get_collection('losses')   # contains classify loss

            # For SGD/Adam, add L2 loss explicitly
            l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            total_loss = tf.add_n(losses) + l2_loss
            
            # Accuracy
            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)

            # Learning rate with decay (same schedule as original)
            DECAY_STEP = 200000
            DECAY_RATE = 0.7
            lr = tf.train.exponential_decay(
                learning_rate,
                batch * BATCH_SIZE,
                DECAY_STEP,
                DECAY_RATE,
                staircase=True
            )
            lr = tf.maximum(lr, 1e-5)

            # Optimizer
            if optimizer_name == "SGD":
                if momentum is not None:
                    optimizer = tf.train.MomentumOptimizer(lr, momentum=momentum)
                else:
                    optimizer = tf.train.GradientDescentOptimizer(lr)
            elif optimizer_name == "Adam":
                optimizer = tf.train.AdamOptimizer(lr)
            else:
                raise ValueError("Unknown optimizer: {}".format(optimizer_name))

            train_op = optimizer.minimize(total_loss, global_step=batch)

            # Summaries
            tf.summary.scalar('total_loss', total_loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('learning_rate', lr)
            merged = tf.summary.merge_all()

            # Saver
            saver = tf.train.Saver()

            # Session
            config_proto = tf.ConfigProto()
            config_proto.gpu_options.allow_growth = True
            config_proto.allow_soft_placement = True
            sess = tf.Session(config=config_proto)
            sess.run(tf.global_variables_initializer())

            ops = {
                'pointclouds_pl': pointclouds_pl,
                'labels_pl': labels_pl,
                'is_training_pl': is_training_pl,
                'pred': pred,
                'loss': total_loss,
                'accuracy': accuracy,
                'train_op': train_op,
                'merged': merged,
                'step': batch,
                'end_points': end_points,
            }

            # Writers (per trial)
            trial_log_dir = OUTPUT_DIR / "trial_{}".format(trial.number)
            trial_log_dir.mkdir(exist_ok=True)
            train_writer = tf.summary.FileWriter(str(trial_log_dir / "train"), sess.graph)
            test_writer = tf.summary.FileWriter(str(trial_log_dir / "test"), sess.graph)

            # ---- Training loop ----
            best_val_macc = 0.0
            epoch_pbar = tqdm(range(MAX_EPOCH), desc="Trial {}".format(trial.number))

            for epoch in epoch_pbar:
                train_loss, train_oa, train_macc = train_one_epoch(
                    sess, ops, train_writer, epoch,
                    TRAIN_DATASET, NUM_CLASSES, BATCH_SIZE
                )

                val_macc = eval_one_epoch(
                    sess, ops, test_writer, epoch,
                    TEST_DATASET, NUM_CLASSES, BATCH_SIZE
                )

                trial.report(val_macc, epoch)
                if trial.should_prune():
                    sess.close()
                    raise optuna.exceptions.TrialPruned()

                if val_macc > best_val_macc:
                    best_val_macc = val_macc
                    save_path = MODEL_DIR / "trial_{}_best.ckpt".format(trial.number)
                    saver.save(sess, str(save_path))

                epoch_pbar.set_postfix_str(
                    "Val mAcc={:.4f} (best={:.4f})".format(val_macc, best_val_macc)
                )

            sess.close()
            return best_val_macc

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=SEED)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=30
    )
    study = optuna.create_study(
        direction="maximize",
        study_name="tf_tuning_{}".format(config),
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///{}".format(DB_PATH),
        load_if_exists=True
    )

    remaining = max(0, TARGET_TRIALS - len(study.trials))
    if remaining > 0:
        print("Existing trials: {} | Running: {}".format(len(study.trials), remaining))
        study.optimize(objective, n_trials=remaining)
    else:
        print("Already have {} trials. Nothing to run.".format(len(study.trials)))

    # Save best results
    best_trial = study.best_trial
    print("\n" + "=" * 80)
    print("BEST TRIAL")
    print("=" * 80)
    print("Best validation mAcc: {:.4f}".format(best_trial.value))
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    with open(str(OUTPUT_DIR / "best_results.txt"), "w") as f:
        f.write("Best validation mAcc: {}\n\n".format(best_trial.value))
        for key, value in best_trial.params.items():
            f.write("{}: {}\n".format(key, value))

    df = study.trials_dataframe()
    df.to_csv(str(OUTPUT_DIR / "all_trials.csv"), index=False)

    print("\n" + "=" * 80)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 80)
    print("\nResults saved to: {}".format(OUTPUT_DIR))
    print("Database: {}".format(DB_PATH))
    print("\nLaunch dashboard:\noptuna-dashboard sqlite:///{}".format(DB_PATH))