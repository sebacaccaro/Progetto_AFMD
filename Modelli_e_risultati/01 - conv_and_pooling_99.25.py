# 187/187 [==============================] - 11s 61ms/step - loss: 0.0026 - accuracy: 0.9995 - val_loss: 0.0402 - val_accuracy: 0.9925

model = models.Sequential()

# We can think filters as the height of the result tensor, while kernel size in the cube we
# use convolve the data

model.add(layers.Conv2D(filters=5, kernel_size=5, strides=1,
                        activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=8, kernel_size=5,
                        strides=1, activation='relu', padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=12, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=15, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling2D())
# ora e' abbastanza overffittoso: piu tardi sperimemto con max pooling e dimensione delle immagini
model.add(layers.Conv2D(filters=18, kernel_size=3, strides=1, activation='relu'))
# model.add(layers.Conv2D(filters=22, kernel_size=3, strides=1, activation='relu')) #senza questo ho 98.7
model.add(layers.Flatten())
# TODO: check if relu it's the loss function we should be using
model.add(layers.Dense(64, activation='relu'))
# Perchè sotmax?
model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
model.summary()
