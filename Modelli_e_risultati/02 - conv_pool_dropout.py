model = models.Sequential()

# We can think filters as the height of the result tensor, while kernel size in the cube we
# use convolve the data

model.add(layers.Conv2D(filters=5, kernel_size=5, strides=1, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH, 3), padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(.2, input_shape=(2,)))
model.add(layers.Conv2D(filters=8, kernel_size=5, strides=1, activation='relu', padding='same'))
model.add(layers.MaxPooling2D())
model.add(layers.Dropout(.2, input_shape=(2,)))
#model.add(layers.Conv2D(filters=12, kernel_size=3, strides=1, activation='relu'))
#model.add(layers.MaxPooling2D())
#model.add(layers.Dropout(.2, input_shape=(2,)))
model.add(layers.Conv2D(filters=15, kernel_size=3, strides=1, activation='relu'))
model.add(layers.MaxPooling2D())
#model.add(layers.Dropout(.2, input_shape=(2,)))
#model.add(layers.Conv2D(filters=18, kernel_size=3, strides=1, activation='relu')) # ora e' abbastanza overffittoso: piu tardi sperimemto con max pooling e dimensione delle immagini
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu')) #TODO: check if relu it's the loss function we should be using
model.add(layers.Dense(len(CLASS_NAMES),activation="softmax"))  #Perchè sotmax?
model.summary()


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, # TODO: We should also tweak with the learing rate
              loss='categorical_crossentropy',
              metrics=['accuracy']) 
