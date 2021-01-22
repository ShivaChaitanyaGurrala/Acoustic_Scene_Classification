import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
from tensorflow.keras.optimizers import RMSprop

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# 1 - Load the model and its pretrained weights if exists
# classifier = cnn()
# classifier.load('weights/cnn_DF')
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Ten neurons as we get the probabilities of all the ten classes
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])
# let us use ImageDataGenerator to pick data to be trained
train_path = pathlib.Path.cwd().parent / 'train'
test_path = pathlib.Path.cwd().parent / 'test'
train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=128,
    class_mode='categorical'
)
test_datagen = ImageDataGenerator(rescale=1 / 255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=128,
    class_mode='categorical'
)
history = model.fit(
    train_generator,
    steps_per_epoch=72,
    epochs=10,
    verbose=1,
    validation_data=test_generator,
    validation_steps=33
)

model.save('newweights/Conv5.h5')
