import os

import keras
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Chargement des données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")

# Prétraitement des données
X_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Modèle
model = Sequential()
model.add(keras.Input(shape=(28, 28, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
model.summary()

# Entraînement
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
model.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_split=0.1, callbacks=[early_stop])

# Évaluation
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nPrécision sur le jeu de test : {acc * 100:.2f}%")

# Sauvegarde
os.makedirs("model", exist_ok=True)
model.save("model/mnist_model.keras")
print("Modèle sauvegardé dans model/mnist_model.keras")