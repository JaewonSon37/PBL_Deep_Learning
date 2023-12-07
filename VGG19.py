from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG19
import tensorflow as tf


# Size of image
image_x, image_y = 100, 100
batch_size = 64
train_dir = "chords"


# Keras model definition
def vgg19_model(image_x, image_y):

    """Build and compile a VGG19 model for guitar chord recognition.

    Returns:
        model: Returns compiled TensorFlow / Keras model.
        callbacks_list: Returns list of Keras callbacks, including model checkpoint.
    """

    num_of_classes = 35   # The number of folders to read

    vgg19 = VGG19(include_top = True, weights = 'imagenet', input_shape = (image_x, image_y, 3))   # Load pre-trained VGG19 model

    x = vgg19.output
    x = Dense(num_of_classes, activation = 'softmax')(x)

    model = tf.keras.Model(inputs = vgg19.input, outputs = x)   # Create the model

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])   # Compile the model

    # ModelCheckpoint callback to save the best model during training
    filepath = "guitar_learner.h5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def main():

    """Main function for training the guitar chord recognition model.
    """

    # Image data generator settings
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,   # Convert to the 0-1 range
        validation_split = 0.2,   # Proportion for validation
        fill_mode = 'nearest')

    # Directory settings for training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (image_x, image_y),   # Input image size
        color_mode = "grayscale",
        batch_size = batch_size,   # Number of images from data
        seed = 42,
        class_mode = 'categorical',
        subset = "training")

    # Directory settings for validation data
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (image_x, image_y),
        color_mode = "grayscale",
        batch_size = batch_size,
        seed = 42,
        class_mode = 'categorical',
        subset = "validation")

    # Create and train the model
    model, callbacks_list = vgg19_model(image_x, image_y)
    model.fit(train_generator, epochs = 5, validation_data = validation_generator)
    
    # Evaluate the model on validation data
    scores = model.evaluate_generator(generator = validation_generator, steps = 64)
    print("VGG19 Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('VGG19_guitar_learner.h5')


if __name__ == '__main__':
    main()