import os

import numpy as np
import pytest
import qkeras
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.datasets import mnist
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow.keras.regularizers import l2
from keras import backend as K
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain
import random

from chisel4ml import chisel4ml_server
from chisel4ml import optimize

from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted


from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.lbir_pb2 import LMFEConfig

from preprocess_full.fft_layer import FFTLayerFULL
from preprocess_full.lmfe_layer import LMFELayerFULL

from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer


from preprocess_nodct.fft_layer import FFTLayerNODCT
from preprocess_nodct.lmfe_layer import LMFELayerNODCT


from preprocess_log2.fft_layer import FFTLayerLOG2
from preprocess_log2.lmfe_layer import LMFELayerLOG2


import csv 

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TFDS_DIR = "/home/ratkop/Documents/jure/quantized_speech_commands/vreca/tensorflow_datasets"
CSV_FILE = "/home/ratkop/Documents/jure/quantized_speech_commands/kws_preprocess_test/proposed-mel-gen.csv"


augment = Compose([
    AddGaussianNoise(min_amplitude=1.0, max_amplitude=175.0, p=0.7),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.7),
    Gain(min_gain_in_db=-3, max_gain_in_db=12, p=0.7)
])



def lr_warmup_cosine_decay(global_step,
                           warmup_steps,
                           hold = 0,
                           total_steps=0,
                           start_lr=0.0,
                           target_lr=1e-3):
    # Cosine decay
    learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold) / float(total_steps - warmup_steps - hold)))

    # Target LR * progress of warmup (=1 at the final warmup step)
    warmup_lr = target_lr * (global_step / warmup_steps)

    # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether `global_step < warmup_steps` and we're still holding.
    # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
    if hold > 0:
        learning_rate = np.where(global_step > warmup_steps + hold,
                                 learning_rate, target_lr)
    
    learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate

class WarmupCosineDecay(tf.keras.callbacks.Callback):
    def __init__(self, total_steps=0, warmup_steps=0, start_lr=0.0, target_lr=1e-3, hold=0):

        super(WarmupCosineDecay, self).__init__()
        self.start_lr = start_lr
        self.hold = hold
        self.total_steps = total_steps
        self.global_step = 0
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = self.model.optimizer.lr.numpy()
        self.lrs.append(lr)

    def on_batch_begin(self, batch, logs=None):
        lr = lr_warmup_cosine_decay(global_step=self.global_step,
                                    total_steps=self.total_steps,
                                    warmup_steps=self.warmup_steps,
                                    start_lr=self.start_lr,
                                    target_lr=self.target_lr,
                                    hold=self.hold)
        K.set_value(self.model.optimizer.lr, lr)
## END: adding Warmup CosineDecay 


# function for reading raw audio data 
def audio_data():
    train_ds, info = tfds.load(
        "speech_commands",
        split="train",
        with_info=True,
        shuffle_files=False,
        as_supervised=True,
        data_dir=TFDS_DIR
    )
    val_ds = tfds.load(
        "speech_commands", split="validation", shuffle_files=False, as_supervised=True, data_dir=TFDS_DIR
    )
    test_ds = tfds.load(
        "speech_commands", split="test", shuffle_files=False, as_supervised=True,data_dir=TFDS_DIR
    )

    label_names = []
    for name in info.features["label"].names:
        print(name, info.features["label"].str2int(name))
        label_names = label_names[:] + [name]

    def get_frames(x):
        npads = (32 * 512) - x.shape[0]
        frames = np.pad(x, (0, npads)).reshape([32, 512])
        frames = np.round(((frames / 2**15)) * 2047 * 0.8)
        return frames.reshape(32, 512)

    def get_frames_aug(x,y):
        
        npads = (32 * 512) - x.shape[0]
        x_np = np.expand_dims(x, axis=0)
        x_np = x_np.astype(np.float32) / (2**15)
        if y != 11:
            rand_int = random.randint(0,10)
            if (rand_int <  3):
                x_np = augment(x_np, sample_rate=16000)

        frames_x = np.pad(x_np, ((0,0),(0, npads))).reshape([32, 512])
        #print(frames_x.shape)
        frames = np.round(((frames_x)) * 2047 * 0.8)
        return frames.reshape(32, 512)

    def train_gen():
        return map(
            #lambda x: tuple([get_frames_aug(x[0],float(x[1])), np.array([float(x[1])])]),
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(train_ds),
        )

    def val_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(val_ds),
        )

    def test_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(x[1])])]),
            iter(test_ds),
        )

    train_set = tf.data.Dataset.from_generator(  # noqa: F841
        train_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )

    val_set = tf.data.Dataset.from_generator(  # noqa: F841
        val_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    test_set = tf.data.Dataset.from_generator(  # noqa: F841
        test_gen,
        output_signature=tuple(
            [
                tf.TensorSpec(shape=(32, 512), dtype=tf.float32),
                tf.TensorSpec(shape=(1), dtype=tf.float32),
            ]
        ),
    )
    return [
        train_set,
        val_set,
        test_set,
        label_names,
        len(train_ds),
        len(val_ds),
        len(test_ds),
    ]

def ds_cnn_v2(audio_data,dict_parms):
    train_set = audio_data[0]  # noqa: F841
    val_set = audio_data[1]  # noqa: F841
    test_set = audio_data[2]
    label_names = audio_data[3]
    TRAIN_SET_LENGTH = audio_data[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data[5]  # noqa: F841
    #668/668 [==============================] - 108s 161ms/step - loss: 0.1546 - sparse_categorical_accuracy: 0.9596 - val_loss: 0.4045 - val_sparse_categorical_accuracy: 0.9033
    #39/39 [==============================] - 6s 148ms/step - loss: 0.5615 - sparse_categorical_accuracy: 0.8448
    input_shape = (32, 512)
    print("Input shape:", input_shape)
    print("label names:", label_names)
    num_labels = len(label_names)

    EPOCHS = dict_parms['epochs']  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2.0), int(input_shape[1]/2.0))

    fft_size = dict_parms['fft_size']
    num_frames = dict_parms['num_frames']
    num_mels = dict_parms['num_mels']

    model = tf.keras.models.Sequential(
        [   
            tf.keras.Input(shape=input_shape),
            FFTLayerLOG2(FFTConfig(fft_size=fft_size, num_frames=num_frames, win_fn=np.hamming(fft_size))),
            LMFELayerLOG2(LMFEConfig(fft_size=fft_size, num_frames=num_frames, num_mels=num_mels)),
            ### First group 
            tf.keras.layers.Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Second group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Third group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Fourth group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Fifth group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            #tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(num_labels),
            tf.keras.layers.Softmax()
        ],
	name = "ds-cnn",
    ) 

    model.summary()



    model_name = "ds-cnn_model"
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH + "/" + f'{model_name}.h5', monitor='val_loss', save_best_only=True)

    total_steps = TRAIN_SET_LENGTH/BATCH_SIZE*EPOCHS
    # If not batched
    #total_steps = len(train_set)/config['BATCH_SIZE']*config['EPOCHS']
    # 5% of the steps
    warmup_steps = int(0.05*total_steps)

    lr_callback = WarmupCosineDecay(total_steps=total_steps, 
                             warmup_steps=warmup_steps,
                             hold=int(warmup_steps/2), 
                             start_lr=0.0, 
                             target_lr=1e-3)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=dict_parms["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),

    )

    acc_list = []

    model.fit_generator(
        train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
        steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
        validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
            EPOCHS
        ),  # noqa: E501
        validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
        epochs=EPOCHS,
        verbose=True,
        callbacks=[lr_callback, early_stop_callback],
    )
    
    acc = model.evaluate(x=test_set.batch(BATCH_SIZE), verbose=True)
    
    return [model,acc]


def ds_cnn(audio_data,dict_parms):
    train_set = audio_data[0]  # noqa: F841
    val_set = audio_data[1]  # noqa: F841
    test_set = audio_data[2]
    label_names = audio_data[3]
    TRAIN_SET_LENGTH = audio_data[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data[5]  # noqa: F841
    #668/668 [==============================] - 108s 161ms/step - loss: 0.1546 - sparse_categorical_accuracy: 0.9596 - val_loss: 0.4045 - val_sparse_categorical_accuracy: 0.9033
    #39/39 [==============================] - 6s 148ms/step - loss: 0.5615 - sparse_categorical_accuracy: 0.8448
    input_shape = (32, 512)
    print("Input shape:", input_shape)
    print("label names:", label_names)
    num_labels = len(label_names)

    EPOCHS = dict_parms['epochs']  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2.0), int(input_shape[1]/2.0))

    fft_size = dict_parms['fft_size']
    num_frames = dict_parms['num_frames']
    num_mels = dict_parms['num_mels']

    model = tf.keras.models.Sequential(
        [   
            tf.keras.Input(shape=input_shape),
            FFTLayerNODCT(FFTConfig(fft_size=fft_size, num_frames=num_frames, win_fn=np.hamming(fft_size))),
            LMFELayerNODCT(LMFEConfig(fft_size=fft_size, num_frames=num_frames, num_mels=num_mels)),
            ### First group 
            tf.keras.layers.Conv2D(filters, (10,4), strides=(2,2), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Second group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Third group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Fourth group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            ### Fifth group 
            tf.keras.layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(3,3), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters, (1,1), padding='same', kernel_regularizer=regularizer),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),

            #tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.Dense(num_labels),
            tf.keras.layers.Softmax()
        ],
	name = "ds-cnn",
    ) 

    model.summary()



    model_name = "ds-cnn_model"
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH + "/" + f'{model_name}.h5', monitor='val_loss', save_best_only=True)

    total_steps = TRAIN_SET_LENGTH/BATCH_SIZE*EPOCHS
    # If not batched
    #total_steps = len(train_set)/config['BATCH_SIZE']*config['EPOCHS']
    # 5% of the steps
    warmup_steps = int(0.05*total_steps)

    lr_callback = WarmupCosineDecay(total_steps=total_steps, 
                             warmup_steps=warmup_steps,
                             hold=int(warmup_steps/2), 
                             start_lr=0.0, 
                             target_lr=1e-3)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=dict_parms["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=tf.keras.metrics.SparseCategoricalAccuracy(),

    )

    acc_list = []

    model.fit_generator(
        train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
        steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
        validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
            EPOCHS
        ),  # noqa: E501
        validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
        epochs=EPOCHS,
        verbose=True,
        callbacks=[lr_callback, early_stop_callback],
    )
    
    acc = model.evaluate(x=test_set.batch(BATCH_SIZE), verbose=True)
    
    return [model,acc]


def main():
    audio_data_files = audio_data()
    
    dict_parms = {
        # preprocess params 
        "fft_size": 512,
        "num_frames": 32,
        #LMFE
        "num_mels": 40,
        # Training parameters
        "lr": 0.5e-3,
        "epochs": 15,
        "mean_acc": 0,
        "std_acc": 0
    }
    

    #mel_list = [10,15,20,25,30,35,40]
    mel_list = [10,15,20,25,30,35,40]
    
    # nodct
    rep_dic = []
    for i in range(len(mel_list)):
        dict_parms["num_mels"] = mel_list[i]
        print("Mel: ", mel_list[i])
        acc_list = []
        for j in range(2):
            [model,acc]= ds_cnn(audio_data_files,dict_parms)
            acc_list.append(acc[1])
        acc_np = np.array(acc_list)
        mean_acc = np.mean(acc_np, axis=0)
        std_acc = np.std(acc_np, axis=0)
        dict_parms["mean_acc"] = mean_acc
        dict_parms["std_acc"] = std_acc
        keys = dict_parms.keys()
        rep_dic.append(dict_parms)
        with open(CSV_FILE, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            if(i==0):
                dict_writer.writeheader()
            dict_writer.writerows(rep_dic)
            rep_dic = []


    # log2 
    rep_dic = []
    for i in range(len(mel_list)):
        dict_parms["num_mels"] = mel_list[i]
        print("Mel: ", mel_list[i])
        acc_list = []
        for j in range(2):
            [model,acc]= ds_cnn_v2(audio_data_files,dict_parms)
            acc_list.append(acc[1])
        acc_np = np.array(acc_list)
        mean_acc = np.mean(acc_np, axis=0)
        std_acc = np.std(acc_np, axis=0)
        dict_parms["mean_acc"] = mean_acc
        dict_parms["std_acc"] = std_acc
        keys = dict_parms.keys()
        rep_dic.append(dict_parms)
        with open(CSV_FILE, 'a', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            if(i==0):
                dict_writer.writeheader()
            dict_writer.writerows(rep_dic)
            rep_dic = []


    # Writing the data to a CSV file
   





if __name__ == "__main__":
    main()

