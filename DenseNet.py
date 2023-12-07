from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten
from keras.layers import MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from torchvision.models import densenet121


# Size of image
image_x, image_y = 200, 200
batch_size = 32
train_dir = "chords"


# Keras model definition
def densenet_model(image_x, image_y):

    """Build and compile a DenseNet model for guitar chord recognition.

    Returns:
        model: Returns compiled TensorFlow / Keras model.
        callbacks_list: Returns list of Keras callbacks, including model checkpoint.

    """

    num_of_classes = 35   # The number of folders to read

    densenet = densenet121(include_top = False, weights = 'imagenet', input_shape = (image_x, image_y, 3))   # Load pre-trained DenseNet model

    densenet.trainable = True
    for i in densenet.layers[:528]:
        i.trainable = False
    for i in densenet.layers[525:]:
        i.trainable = True
        print(i.name, i.trainable)

    x = densenet.output
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation = 'relu', input_dim = (200, 200, 3))(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = Dense(num_of_classes, activation = 'softmax')(x)

    model = tf.keras.Model(inputs = densenet.input, outputs = x)   # Create the model

    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])   # Compile the model

    # ModelCheckpoint callback to save the best model during training
    filepath = "guitar_learner.h5"
    checkpoint = ModelCheckpoint(filepath, monitor = 'val_acc', verbose = 1, save_best_only = True, mode = 'max')
    callbacks_list = [checkpoint]

    return model, callbacks_list


def main():
    
    """Main function for training the guitar chord recognition model.
    """

    # Data augmentation settings
    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,   # Convert to the 0-1 range
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        rotation_range = 15,
        zoom_range = 0.2,
        horizontal_flip = False,
        validation_split = 0.2,   # Proportion for validation
        fill_mode = 'nearest')

    # Directory settings for training data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (image_x, image_y),   # Input image size
        color_mode = "rgb",
        batch_size = batch_size,   # Number of images from data
        seed = 42,
        class_mode = 'categorical',
        subset = "training")

    # Directory settings for validation data
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (image_x, image_y),
        color_mode = "rgb",
        batch_size = batch_size,
        seed = 42,
        class_mode = 'categorical',
        subset = "validation")

    # Create and train the model
    model, callbacks_list = densenet_model(image_x, image_y)
    model.fit(train_generator, epochs = 5, validation_data = validation_generator)
    
    # Evaluate the model on validation data
    scores = model.evaluate_generator(generator = validation_generator, steps = 64)
    print("DenseNet Error: %.2f%%" % (100 - scores[1] * 100))

    model.save('DenseNet_guitar_learner.h5')


if __name__ == '__main__':
    main()