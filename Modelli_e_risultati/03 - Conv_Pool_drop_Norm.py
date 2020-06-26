def createAndTrain4():
    history = None
    model = models.Sequential()
    model.add(layers.Conv2D(filters=5, kernel_size=5, strides=1,
                            use_bias=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Conv2D(filters=8, kernel_size=5,
                            strides=1, use_bias=False, padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Conv2D(filters=12, kernel_size=3,
                            strides=1, use_bias=False,  padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Conv2D(filters=15, kernel_size=3,
                            strides=1, use_bias=False, padding='same'))
    model.add(layers.MaxPooling2D())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')
    model.add(layers.Dropout(0.06))

    model.add(layers.Conv2D(filters=18, kernel_size=3,
                            strides=1, use_bias=False, padding='same'))
    model.add(layers.Flatten())
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')

    model.add(layers.Dense(64, use_bias=False))
    model.add(layers.BatchNormalization(center=True, scale=False))
    tf.keras.layers.Activation('relu')
    model.add(layers.Dropout(0.06))

    model.add(layers.Dense(len(CLASS_NAMES), activation="softmax"))
    model.summary()

    print(model.summary())

    opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
