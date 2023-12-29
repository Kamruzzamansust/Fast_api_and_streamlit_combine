import tensorflow as tf 
import streamlit as st
from fastapi import FastAPI
import pandas as pd 
import numpy as np 
import tensorflow as tf### models
import pandas as pd ### reading and processing data
import seaborn as sns ### visualization
import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError 
from tensorflow.keras.optimizers import Adam

data = pd.read_csv('G:\\FAST\\prjects\\combine\\train.csv')

tensor_data = tf.constant(data)
tensor_data = tf.cast(tensor_data, tf.float32)
tensor_data = tf.random.shuffle(tensor_data)
X = tensor_data[:,3:-1]
# # print(X[:5])

y = tensor_data[:,-1]
# #print(y[:5].shape)
y = tf.expand_dims(y, axis = -1)
# #print(y[:5])

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

X_train = X[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE*TRAIN_RATIO)]
# print(X_train.shape)
# print(y_train.shape)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)


X_val = X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
# print(X_val.shape)
# print(y_val.shape)


val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = train_dataset.shuffle(buffer_size = 8, reshuffle_each_iteration = True).batch(32).prefetch(tf.data.AUTOTUNE)


X_val = X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
# print(X_val.shape)
# print(y_val.shape)

X_test = X[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
# print(X_test.shape)
# print(y_test.shape)


normalizer = Normalization()
normalizer.adapt(X_train)
normalizer(X)[:5]



# model = tf.keras.Sequential([
#                              InputLayer(input_shape = (8,)),
#                              normalizer,
#                              Dense(128, activation = "relu"),
#                              Dense(128, activation = "relu"),
#                              Dense(128, activation = "relu"),
#                              Dense(1),
# ])
# model.summary()


# model.compile(optimizer = Adam(learning_rate = 0.1),
#               loss = MeanAbsoluteError(),
#               metrics = RootMeanSquaredError())


# model.predict(tf.expand_dims(X_test[0], axis = 0 ))










# import tensorflow as tf
# from tensorflow.keras.layers import InputLayer, Dense, Normalization
# from tensorflow.keras.losses import MeanAbsoluteError
# from tensorflow.keras.metrics import RootMeanSquaredError
# from tensorflow.keras.optimizers import Adam

# Load data and preprocess
# ... (your data loading and preprocessing code)

# Build the model
model = tf.keras.Sequential([
    InputLayer(input_shape=(8,)),
    Normalization(),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(128, activation="relu"),
    Dense(1),
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.1),
    loss=MeanAbsoluteError(),
    metrics=[RootMeanSquaredError()]
)

# Train the model
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# Save the model
model.save("my_saved_model")

# Example of loading the saved model
# loaded_model = tf.keras.models.load_model("your_model_directory_path")

# Make predictions
predictions = model.predict(tf.expand_dims(X_test[0], axis=0))
print("Predictions:", predictions)























