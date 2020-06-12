model = models.Sequential()

# We can think filters as the height of the result tensor, while kernel size in the cube we
# use convolve the data

model.add(layers.Conv2D(filters=12, kernel_size=3, strides=2, activation='relu', input_shape=(IMG_HEIGHT,IMG_WIDTH, 3)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu')) #TODO: check if relu it's the loss function we should be using
model.add(layers.Dense(len(CLASS_NAMES),activation="softmax"))  #Perchè softmax?
model.summary()



model.compile(optimizer='adam', # TODO: We should also tweak with the learing rate
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              

# utility callback that displays training curves
plot_training = PlotTraining(5550, sample_rate=10, zoom=5) #zoom originale = 16

# lr decay function, reduces the learning rate if it's raising too much
def lr_decay(epoch):
  return 0.01 * math.pow(0.6, epoch)

# lr schedule callback
lr_decay_callback = tf.keras.callbacks.LearningRateScheduler(lr_decay, verbose=True)

history = None



steps_per_epoch = int(image_count/BATCH_SIZE)
EPOCHS = 10           #TODO Probabilmente da aumentare

# https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
history = model.fit(x=train_ds,
                    validation_data=validation_ds,
                    validation_steps=1,
                    epochs=EPOCHS, 
                    batch_size=BATCH_SIZE,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[plot_training]#callbacks=[plot_training, lr_decay_callback]
                    )
                    
                    
                    
'''Results                    
46/46 [==============================] - 8s 183ms/step - loss: 0.1152 - accuracy: 0.9805 - val_loss: 0.1529 - val_accuracy: 0.9550
'''
