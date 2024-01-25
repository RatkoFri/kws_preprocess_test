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

from qkeras.utils import load_qmodel
from audiomentations import Compose, AddGaussianNoise, PitchShift, Gain
import random


from chisel4ml import chisel4ml_server
from chisel4ml import optimize
from chisel4ml.preprocess.fft_layer import FFTLayer
from chisel4ml.preprocess.lmfe_layer import LMFELayer
from chisel4ml.lbir.lbir_pb2 import FFTConfig
from chisel4ml.lbir.lbir_pb2 import LMFEConfig

from preprocess_full.fft_layer import FFTLayerFULL
from preprocess_full.lmfe_layer import LMFELayerFULL

from preprocess_nodct.fft_layer import FFTLayerNODCT
from preprocess_nodct.lmfe_layer import LMFELayerNODCT

from chisel4ml.qkeras_extensions import FlattenChannelwise
from chisel4ml.qkeras_extensions import QDepthwiseConv2DPermuted

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
TFDS_DIR = "/home/ratkop/Documents/jure/quantized_speech_commands/vreca/tensorflow_datasets"
MODEL_PATH = "/home/ratkop/Documents/jure/quantized_speech_commands/teacher_models/"
TEACHER_MODEL = "teacher_model_final_3classes_with_uknown_weight_class1.h5"
CSV_FILE = "/home/ratkop/Documents/jure/quantized_speech_commands/csv/final-distiled-nclasses.csv"

augment = Compose([
    AddGaussianNoise(min_amplitude=1.0, max_amplitude=175.0, p=0.7),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.7),
    Gain(min_gain_in_db=-3, max_gain_in_db=12, p=0.7)
])


def filter_classes(y, allowed_classes: list):
    """
    Returns `True` if `y` belongs to `allowed_classes` list else `False`
    Example usage:
        dataset.filter(lambda s: filter_classes(s['label'], [0,1,2])) # as dict
        dataset.filter(lambda x, y: filter_classes(y, [0,1,2])) # as_supervised
    """
    allowed_classes = tf.constant(allowed_classes)
    isallowed = tf.equal(allowed_classes, tf.cast(y, allowed_classes.dtype))
    reduced_sum = tf.reduce_sum(tf.cast(isallowed, tf.float32))
    return tf.greater(reduced_sum, tf.constant(0.))

# function for reading raw audio data 
def audio_data(classes):
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

    # number of samples per class 


    train_ds = train_ds.filter(lambda x, y: filter_classes(y, classes))
    val_ds   = val_ds.filter(lambda x, y: filter_classes(y, classes))
    test_ds  = test_ds.filter(lambda x, y: filter_classes(y, classes))

    label_names = []
    for name in info.features["label"].names:
        if info.features["label"].str2int(name) in classes:
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
        # y = 2 is unknown class
        if y != 2:
            rand_int = random.randint(0,10)
            if (rand_int <  3):
                x_np = augment(x_np, sample_rate=16000)

        frames_x = np.pad(x_np, ((0,0),(0, npads))).reshape([32, 512])
        return frames_x.reshape(32, 512)

    def train_gen():
        return map(
            lambda x: tuple([get_frames_aug(x[0],float(classes.index(x[1]))), np.array([float(classes.index(x[1]))])]),
            iter(train_ds),
        )

    def val_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(classes.index(x[1]))])]),
            iter(val_ds),
        )

    def test_gen():
        return map(
            lambda x: tuple([get_frames(x[0]), np.array([float(classes.index(x[1]))])]),
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
    
    train_xx = list(train_ds.as_numpy_iterator())
    val_xx = list(val_ds.as_numpy_iterator())
    test_xx = list(test_ds.as_numpy_iterator())



    return [
        train_set,
        val_set,
        test_set,
        label_names,
        len(train_xx),
        len(val_xx),
        len(test_xx),
    ]

def ds_cnn(audio_data,dict_params):
    train_set = audio_data[0]  # noqa: F841
    val_set = audio_data[1]  # noqa: F841
    test_set = audio_data[2]
    label_names = audio_data[3]
    TRAIN_SET_LENGTH = audio_data[4]  # noqa: F841
    VAL_SET_LENGTH = audio_data[5]  # noqa: F841
    #668/668 [==============================] - 108s 161ms/step - loss: 0.1546 - sparse_categorical_accuracy: 0.9596 - val_loss: 0.4045 - val_sparse_categorical_accuracy: 0.9033
    #39/39 [==============================] - 6s 148ms/step - loss: 0.5615 - sparse_categorical_accuracy: 0.8448
    input_shape = (32,512)
    print("Input shape:", input_shape)
    print("label names:", label_names)
    num_labels = len(label_names)

    EPOCHS = 20  # noqa: F841
    BATCH_SIZE = 128  # noqa: F841

    filters = 64
    weight_decay = 1e-4
    regularizer = l2(weight_decay)
    final_pool_size = (int(input_shape[0]/2.0), int(input_shape[1]/2.0))

    model = tf.keras.models.Sequential(
        [   
            tf.keras.layers.Input(shape=input_shape),
            FFTLayerFULL(FFTConfig(fft_size=dict_params['fft_size'], num_frames=dict_params['num_frames'], win_fn=np.hamming(dict_params['fft_size']))),
            LMFELayerFULL(LMFEConfig(fft_size=dict_params['fft_size'], num_frames=dict_params['num_frames'], num_mels=dict_params['num_mels'])),
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

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=dict_params["lr"]),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    opt_model = model
    opt_model.summary()
    

    total_steps = TRAIN_SET_LENGTH/BATCH_SIZE*EPOCHS
    # If not batched
    #total_steps = len(train_set)/config['BATCH_SIZE']*config['EPOCHS']
    # 5% of the steps
    warmup_steps = int(0.05*total_steps)




    reduce_LR = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=dict_params["lr"]/1000,)

    opt_model.fit_generator(
        train_set.batch(BATCH_SIZE, drop_remainder=True).repeat(EPOCHS),  # noqa: E501
        steps_per_epoch=int(TRAIN_SET_LENGTH / BATCH_SIZE),
        validation_data=val_set.batch(BATCH_SIZE, drop_remainder=True).repeat(
            EPOCHS
        ),  # noqa: E501
        validation_steps=int(VAL_SET_LENGTH / BATCH_SIZE),
        epochs=EPOCHS,
        verbose=True,
        callbacks=[reduce_LR],
    )

    #opt_model.save( MODEL_PATH + TEACHER_MODEL)

    acc = opt_model.evaluate(x=test_set.batch(BATCH_SIZE), verbose=True)
    return [opt_model,acc]    


def main():
    classes = [0,1,2,3,4,5,6,7,8,9,10,11]
    audio_data_files = audio_data(classes)

    dict_parms = {
        # preprocess params 
        "fft_size": 512,
        "num_frames": 32,
        #LMFE
        "num_mels": 20,
        # Training parameters
        "lr": 1e-3,
        "epochs": 20,
        "acc": 0,
    }
    
    [model,acc]= ds_cnn(audio_data_files,dict_parms)
    print("Accuracy: ",acc)
    
if __name__ == "__main__":
    main()

