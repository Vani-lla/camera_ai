import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

from helpers import get_data

X, Y = get_data()

global_scaller = StandardScaler()
X = global_scaller.fit_transform(X)

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=X[0].shape),
    tf.keras.layers.Dense(128),
    tf.keras.layers.Dense(5)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(X, Y, epochs=50)

model.save("models/model.h5")