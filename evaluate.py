import argparse
import pickle
import modelnet_data
import pointhop
import numpy as np
import data_utils
import os
import time
import sklearn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--initial_point', type=int, default=1024, help='Point Number [256/512/1024/2048]')
parser.add_argument('--validation', default=False, help='Split train data or not')
parser.add_argument('--feature_selection', default=0.95, help='Percentage of feature selection try 0.95')
parser.add_argument('--ensemble', default=True, help='Ensemble or not')
parser.add_argument('--rotation_angle', default=np.pi/4, help='Rotate angle')
parser.add_argument('--rotation_freq', default=8, help='Rotate time')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', default=[1024, 128, 128, 64], help='Point Number after down sampling')
parser.add_argument('--num_sample', default=[64, 64, 64, 64], help='KNN query number')
FLAGS = parser.parse_args()

initial_point = FLAGS.initial_point
VALID = FLAGS.validation
FE = FLAGS.feature_selection
ENSEMBLE = FLAGS.ensemble
angle_rotation = FLAGS.rotation_angle
freq_rotation = FLAGS.rotation_freq
num_point = FLAGS.num_point
num_sample = FLAGS.num_sample


LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_eva.txt'), 'w')
LOG_FOUT.write(str(FLAGS) + '\n')


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def main():
    time_start = time.time()
    # load data
    train_data, train_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=True)
    test_data, test_label = modelnet_data.data_load(num_point=initial_point, data_dir=os.path.join(BASE_DIR, 'modelnet40_ply_hdf5_2048'), train=False)

    # validation set
    if VALID:
        train_data, train_label, valid_data, valid_label = modelnet_data.data_separate(train_data, train_label)
    else:
        valid_data = test_data
        valid_label = test_label

    print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)

    if ENSEMBLE:
        angle = np.repeat(angle_rotation, freq_rotation)
    else:
        angle = [0]

    feat_valid = []
    for i in range(len(angle)):
        with open(os.path.join(LOG_DIR, 'params.pkl'), 'rb') as f:
            params_total = pickle.load(f)
        params = params_total['params:', i]
        weight = params_total['weight:', i]

        log_string('------------Test {} --------------'.format(i))
        leaf_node_test = pointhop.pointhop_pred(False, valid_data, pca_params=params, n_newpoint=num_point, n_sample=num_sample)
        feature_valid = pointhop.extract(leaf_node_test)
        feature_valid = np.concatenate(feature_valid, axis=-1)
        if FE is not None:
            fe_ind = params_total['fe_ind:', i]
            feature_valid = feature_valid[:, fe_ind]
        feature_valid, pred_valid = pointhop.llsr_pred(feature_valid, weight)
        feat_valid.append(feature_valid)

        acc_valid = sklearn.metrics.accuracy_score(valid_label, pred_valid)
        acc = pointhop.average_acc(valid_label, pred_valid)
        log_string('test: {} , test mean: {}'.format(acc_valid, np.mean(acc)))
        log_string('per-class: {}'.format(acc))
        valid_data = data_utils.data_augment(valid_data, angle[i])

    if ENSEMBLE:
        weight = params_total['weight ensemble']
        feat_valid = np.concatenate(feat_valid, axis=-1)
        feat_valid, pred_valid = pointhop.llsr_pred(feat_valid, weight)
        acc_valid = sklearn.metrics.accuracy_score(valid_label, pred_valid)
        acc = pointhop.average_acc(valid_label, pred_valid)
        log_string('ensemble test: {}, ensemble test mean: {}'.format(acc_valid, np.mean(acc)))
        log_string('ensemble per-class: {}'.format(acc))

    time_end = time.time()
    log_string('totally time cost is {} minutes'.format((time_end - time_start)//60))


if __name__ == '__main__':
    main()