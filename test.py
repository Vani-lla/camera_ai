import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
print("TensorFlow version:", tf.__version__)
mnist = tf.keras.datasets.mnist

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train, y_train = [], []
for i in range(5):
    tmp = np.load(f"training_data/{i}.npy")
    x_train.append(tmp)
    for _ in range(len(tmp)):
        y_train.append(i)

x_train = np.row_stack(x_train).reshape(-1, 21*2*20)
y_train = np.array(y_train)

print(x_train)
print(x_train.shape)

scaller = StandardScaler()
x_train = scaller.fit_transform(x_train)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=x_train[0].shape),
  tf.keras.layers.Dense(128),
  tf.keras.layers.Dense(5)
])

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50)

model.evaluate(x_train,  y_train, verbose=2)
x_test: np.ndarray = np.load("test.npy")
print(x_test.shape)
x_test = np.array(list(x.flatten() for x in x_test))

print(x_test)
print(x_test.shape)
print(np.argmax(model.predict(x_test), axis=1))


checkpoint_path = "model.h5"
model.save(checkpoint_path)
