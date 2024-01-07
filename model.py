# model.py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=8, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, training_set, test_set, epochs=25):
    model.fit(training_set, steps_per_epoch=len(training_set), epochs=epochs, validation_data=test_set,
              validation_steps=len(test_set))
    return model
