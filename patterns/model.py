import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.learn.python.learn as skflow
from PIL import Image
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA, PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.contrib.learn.python.learn import monitors
from config import ROOT_PATH
import datetime as dt
import pandas.io.data as web


STANDARD_SIZE = (50, 50)
tf.logging.set_verbosity(tf.logging.INFO)


def get_image_data(filename):
    img = Image.open(filename)
    img = img.getdata()
    img = img.resize(STANDARD_SIZE)
    img = map(list, img)
    img = np.array(img)
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)
    return img_wide[0]


def load_data(kind='train'):
    data = []
    labels = []
    key = {}
    i = 0

    if kind == 'train':
        # load targets
        files = [f for f in os.listdir(ROOT_PATH + 'template/')]
        for f in files:
            data.append(get_image_data(ROOT_PATH + 'template/' + f))
            labels.append(1)
            key[i] = f
            i += 1

        # load random plots
        files = [f for f in os.listdir(ROOT_PATH + 'random/')]
        for f in files:
            data.append(get_image_data(ROOT_PATH + 'random/' + f))
            labels.append(0)
            key[i] = f
            i += 1

    elif kind == 'predict':
        files = [f for f in os.listdir(ROOT_PATH + 'current/')]
        for f in files:
            data.append(get_image_data(ROOT_PATH + 'current/' + f))
            labels.append(0)
            key[i] = f
            i += 1

    print "Data has been loaded"
    return data, labels, key


def save_predictions(predictions, data_dict, file_label, key_pairs,
                     kind='train'):
    """

    :param predictions:
    :param file_label:
    :param key_pairs:
    :param kind:
    :return:

    >>> predictions = _pred
    >>> data_dict = data_dict
    >>> file_label = 'svm_predict'
    >>> key_pairs = key_p
    >>> kind='predict'
    predictions=_pred_p; data_dict=data_dict_p; key_pairs=key_p
    file_label='svm_predict'; kind='predict'

    predictions=_pred ; data_dict=data_dict; key_pairs=key
    file_label = 'tf_train'; kind='train'
    """
    # Check out these plots:

    if kind == 'train':
        folder = 'random/'
        save_dir = 'train/'
        mis_classify = data_dict['y_test'][data_dict['y_test'] != predictions] == 0
        _f = [key_pairs[idx] for idx, x in mis_classify.iteritems() if x == True]

    elif kind == 'predict':
        folder = 'current/'
        save_dir = 'prediction/'
        mis_classify = data_dict['y_test'][data_dict['y'] != predictions] == 1
        _f = [key_pairs[idx] for idx, x in mis_classify.iteritems()]

    for file_name in _f:
        img = Image.open(ROOT_PATH + folder + file_name)
        img.save(ROOT_PATH + 'results/' + save_dir + file_label + "_" + file_name)

    if len(_f) == 0:
        print "No Results"
        print "Here is the confusion matrix: "
        print confusion_matrix(data_dict['y'], predictions)
    else:
        print "Saved results to /results/".format(ROOT_PATH + folder)


def preprocessing(data, labels, pca=False, scale=False):
    data_dict = {}
    X = np.array(data)
    _labels = pd.Series(labels)
    X_train, X_test, y_train, y_test = train_test_split(np.array(data), _labels,
                                                        test_size=0.4, random_state=2)

    if pca:
        pca = PCA(n_components=100)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        X = pca.transform(X)

    if scale:
        std_scaler = StandardScaler()
        X_train = std_scaler.fit_transform(X_train)
        X_test = std_scaler.transform(X_test)
        X = std_scaler.transform(X)

    # return np.array(X_train), np.array(X_test), y_train, y_test
    data_dict['X_test'] = np.array(X_test)
    data_dict['X_train'] = np.array(X_train)
    data_dict['y_train'] = y_train
    data_dict['y_test'] = y_test
    data_dict['X'] = X
    data_dict['y'] = _labels

    return data_dict


def train_simple_svm_model(data_dict):
    # Higher C seems to help with the desired overfitting
    # PCA speeds up a lot and helps with false predictions at ~500
    clf = SVC(C=10000000, gamma=0.00001)
    clf.fit(data_dict['X_train'], data_dict['y_train'])

    _pred = clf.predict(data_dict['X_test'])
    print accuracy_score(_pred, data_dict['y_test'].values)
    print confusion_matrix(_pred, data_dict['y_test'])

    print "FINISHED SVM"
    return clf


def train_inital_tf_model(data):
    """ Run this only once, otherwise it take forever    """
    layers, steps, lr = [32, 128], 10000, .01
    layers_str = str(layers).replace('[', '').replace(']', '').replace(', ', '_')
    log_dir = '/tmp/tf_examples/{}_DNN_{}_{}_{}/'.format("two_layer_final_model", layers_str, steps, lr)

    model = skflow.TensorFlowDNNClassifier(
            hidden_units=layers,
            n_classes=2,
            batch_size=128,
            steps=steps,
            learning_rate=lr)
    m = monitors.ValidationMonitor(data['X_train'], data['y_train'].values,
                                   every_n_steps=200)
    model.fit(data['X_train'], list(data['y_train'].values),
              logdir=log_dir, monitors=[m])

    _pred = model.predict(data['X'])
    print accuracy_score(_pred, data['y'].values)
    print confusion_matrix(_pred, data['y'])
    # model.save(log_dir)

    return model, log_dir

def get_updated_model(log_file, data):
    """
    :param name:
    :param data:
    :param labels:
    :return:

    >>> log_file = '/tmp/tf_examples/two_layer_final_model_DNN_32_128_10000_0.01/'
    >>> data  = data_dict_p
    """
    layers, steps, lr = [32, 128], 10000, .01
    model = skflow.TensorFlowDNNClassifier(
            hidden_units=layers,
            n_classes=2,
            batch_size=128,
            steps=steps,
            learning_rate=lr)

    m = monitors.ValidationMonitor(data['X_train'], data['y_train'].values,
                                   every_n_steps=200)
    model.fit(data['X'], list(data['y'].values),
              logdir=log_file)

    _pred = model.predict(data['X'])
    print accuracy_score(_pred, data['y'].values)
    print confusion_matrix(_pred, data['y'])

    return model


def get_best_chart(force_models_agree=True):
    # TODO need logic to prevent picking the same chart multiple times
    # if svm and dnn agree, choose that chart, otherwise, pick randomly.
    files = [f for f in os.listdir(ROOT_PATH + '/results/prediction/')]
    tf_files = [x.replace('tf_predict_random_plots_',"") for x in files if x[:2]=='tf']
    svm_files = [x.replace('svm_predict_random_plots_',"") for x in files if x[:3]=='svm']
    if force_models_agree:
        common_charts = [x for x in tf_files if x in svm_files]
    common_charts = svm_files + tf_files

    # remove already tweeted securities.
    posted_plots = [x[0:3] for x in os.listdir(ROOT_PATH + '/results/posted_plots/')]
    common_charts = [x for x in common_charts if x[0:3] not in posted_plots]

    if len(common_charts) >0:
        selection = common_charts[np.random.randint(0, len(common_charts))]
        print "Selecting {}".format(selection)
        ticker, start, end = selection.replace('.jpg',"").split("_")
        prices = web.DataReader(ticker, 'yahoo', dt.datetime(2014, 1, 1), dt.date.today())

        return ticker, prices['2014-1-1':].iloc[int(start):int(end)][['Open','High','Low','Close']]
    else:
        return

def build_predictions():
    # run the training data set
    data, labels, key = load_data(kind='train')

    # train SVM since its fast
    data_dict_svm = preprocessing(data, labels, True, True)
    assert(data_dict_svm['X'].shape[1] >= 100), "Number of observations must be greater than 100, add more tickers"
    svm_model = train_simple_svm_model(data_dict_svm)

    # Train TF model
    data_dict_tf = preprocessing(data, labels, False, True)
    tf_model, tf_model_log = train_inital_tf_model(data_dict_tf)

    # load the prediction data set
    data_p, labels_p, key_p = load_data(kind='predict')

    # Find current HS to sent out using SVM
    data_dict_p_svm = preprocessing(data_p, labels_p, pca=True, scale=True)
    assert(data_dict_p_svm['X'].shape[1] >= 100),"Number of prediction observations must be greater than 100, add more tickers"
    svm_pred_p = svm_model.predict(data_dict_p_svm['X'])
    save_predictions(predictions=svm_pred_p, data_dict=data_dict_p_svm, key_pairs=key_p,
                     file_label='svm_predict', kind='predict')

    # predict using TF
    data_dict_p = preprocessing(data_p, labels_p, pca=False, scale=True)
    tf_model = get_updated_model(tf_model_log, data_dict_p)
    nn_pred_p = tf_model.predict(data_dict_p['X'])
    save_predictions(predictions=nn_pred_p, data_dict=data_dict_p, key_pairs= key_p,
                     file_label='tf_predict', kind='predict')


if __name__ == "__main__":
    # Some resources:
    # https://github.com/tensorflow/skflow/blob/master/g3doc/get_started/index.md
    # https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/learn/python/learn
    # tensorboard --logdir=/tmp/tf_examples/

    pass