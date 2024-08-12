import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
import numpy as np

from helpers import get_data

num = 4

X, Y = get_data('left', num)
Y += 1

# global_scaller = StandardScaler()
# X = global_scaller.fit_transform(X)
# print(X[0].shape)
# print(X.shape)
# print(X[0])
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(63),
    tf.keras.layers.Dense(21),
    tf.keras.layers.Dense(num + 1, activation="softmax")
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

model.fit(X, Y, epochs=25)

model.save("models/lhm3d.keras")
print(model(X[15].reshape(-1, 63)))