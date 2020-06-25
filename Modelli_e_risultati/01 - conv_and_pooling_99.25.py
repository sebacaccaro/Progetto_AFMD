model = models.Sequential()
model.add(layers.Conv2D(filters=5, kernel_size=5, strides=1,
                        activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=8, kernel_size=5,
                        strides=1, activation='relu', padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=12, kernel_size=3,
                        strides=1, activation='relu', padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=15, kernel_size=3,
                        strides=1, activation='relu', padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(filters=18, kernel_size=3,
                        strides=1, activation='relu', padding='same'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
model.summary()
