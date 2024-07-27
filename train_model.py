import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np

from helpers import get_data

num = 4

X, Y = get_data('left', num)
Y += 1

# global_scaller = StandardScaler()
# X = global_scaller.fit_transform(X)
print(X[0].shape)
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=X[0].shape),
    tf.keras.layers.Dense(21),
    tf.keras.layers.Dense(num + 1)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(X, Y, epochs=50)

model.save("models/lhm3d.h5")
