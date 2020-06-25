# VAL_ACCURACY 98.75% - ACCURACY 100%


model = models.Sequential()

# We can think filters as the height of the result tensor, while kernel size in the cube we
# use convolve the data

model.add(layers.Conv2D(filters=5, kernel_size=5, strides=3, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH, 3), padding='same'))
model.add(layers.Conv2D(filters=8, kernel_size=5, strides=2, activation='relu', padding='same'))
model.add(layers.Conv2D(filters=12, kernel_size=3, strides=1, activation='relu'))
model.add(layers.Conv2D(filters=15, kernel_size=3, strides=1, activation='relu'))
model.add(layers.Conv2D(filters=18, kernel_size=3, strides=1, activation='relu')) # ora e' abbastanza overffittoso: piu tardi sperimemto con max pooling e dimensione delle immagini
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu')) #TODO: check if relu it's the loss function we should be using
model.add(layers.Dense(len(CLASS_NAMES),activation="softmax"))  #Perchè sotmax?
model.summary()



model.compile(optimizer='adam', # TODO: We should also tweak with the learing rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])
 
