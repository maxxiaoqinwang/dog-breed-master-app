"""author: Max Wang
   date: 28/01/2018"""

print('Loading libraries and initializing ...')

from keras.applications.resnet50 import ResNet50
import pandas as pd
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import urllib.request
from extract_bottleneck_features import *
from tqdm import tqdm

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import warnings
warnings.filterwarnings("ignore")

# load dog breeds
df_dog_breed = pd.read_csv('dog_name.csv')
dog_names = df_dog_breed.values

# load ResNet50 model
ResNet50_model = ResNet50(weights='imagenet')

def ResNet50_predict_labels(tensor: np.array) -> int:
    """Using the established ResNet50 to identify the contain of the image.
    """
    # returns prediction vector for image located at img_path
    img = preprocess_input(tensor)
    return np.argmax(ResNet50_model.predict(img))


# whether there is a dog in the image
def dog_detector(tensor: np.array) -> bool:
    """Using Established ResNet50 to identify whether it is a dog image"""
    prediction = ResNet50_predict_labels(tensor)
    # ImageNet categories: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
    # 151 to 269 are dogs in ImageNet, if the predicted label is in this range
    # return True
    return (prediction <= 268) & (prediction >= 151)


# whether there is a cat in the image
def cat_detector(tensor: np.array) -> bool:
    """Using Established ResNet50 to identify whether it is a dog image"""
    prediction = ResNet50_predict_labels(tensor)
    # 281 to 287 are dogs in ImageNet, if the predicted label is in this range
    # return True
    return (prediction <= 287) & (prediction >= 281)


def path_to_tensor(img_path: str, is_url=False) -> np.array:
    """Transfer image to 4D tensor.
    Args:
        img_path: image location.
        is_url: whether the path is an url.

    Return:
        4D tensor of the image."""
    if is_url:
        urllib.request.urlretrieve(img_path, 'img_buf.jpg')
        img_path = 'img_buf.jpg'
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert image to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)

    if is_url:
        os.remove('img_buf.jpg')
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3)
    return np.expand_dims(x, axis=0)


def load_model():
    """Create the model structure.
    Return:
        keras model."""
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_shape=(6144,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(133, activation='softmax'))
    model.load_weights('saved_models/weights.best.model1.hdf5')
    return model


def model_predict_breed(model: Sequential, img_path: str, is_url=False):
    """The function to predict dog breed.
    Args:
        model: the redefined keras model.
        img_path: image location.
        is_url: whether the path location is an url."""

    try:
        keras_tensor = path_to_tensor(img_path, is_url)
    except Exception as error:
        print('Hmm, something looks wrong: ' + repr(error))
        return

    if dog_detector(keras_tensor):
        print('Oh, it\'s a cute dog! Let me check out the breed ...')
    else:
        if cat_detector(keras_tensor):
            print('I hate cats! Take it away from me!')
        else:
            print('It is not a dog!')
        return

    Resnet50_feature = extract_Resnet50(keras_tensor)
    print('Let me have a look at the coat ...')
    InceptionV3_feature = extract_InceptionV3(keras_tensor)
    print('Let me measure the ear length ...')
    Xception_feature = extract_Xception(keras_tensor)
    print('Let me measure the nose size ...')

    InceptionV3_flatten_model = Sequential()
    InceptionV3_flatten_model.add(GlobalAveragePooling2D(input_shape=InceptionV3_feature.shape[1:]))
    Xception_flatten_model = Sequential()
    Xception_flatten_model.add(GlobalAveragePooling2D(input_shape=Xception_feature.shape[1:]))
    Resnet50_flatten_model = Sequential()
    Resnet50_flatten_model.add(GlobalAveragePooling2D(input_shape=Resnet50_feature.shape[1:]))

    InceptionV3_feature = InceptionV3_flatten_model.predict(InceptionV3_feature)
    Xception_feature = Xception_flatten_model.predict(Xception_feature)
    Resnet50_feature = Resnet50_flatten_model.predict(Resnet50_feature)

    feature = np.concatenate((InceptionV3_feature, Xception_feature, Resnet50_feature), axis=1)
    print('OK. I get some information on this dog. Let me look into my Dog Breed Encyclopaedia ...')
    print('It may take awhile ...\n')

    # create prediction vactor
    predicted_vector = model.predict(feature)
    # return dog breed that has the maximum likelihood
    print('According to my encyclopaedia, it\'s a/an {}.\n'.format(dog_names[np.argmax(predicted_vector)][0]))
    return


if __name__ == "__main__":
    model = load_model()
    os.system('cls')

    print('Hello! I am a dog breed master. Show me a real dog image. I will tell you its breed.')
    while True:
        img_path = input('Please type the path to the image:\n')
        while True:
            url_bool = input('Is it on Internet? [Y/N]\n')
            if url_bool == 'Y':
                is_url = True
                break
            elif url_bool == 'N':
                is_url = False
                break
            else:
                print('You can only choose from [Y/N].')

        model_predict_breed(model, img_path, is_url)

        again_bool = input('Have another try? [Y/N]')
        if again_bool != 'Y':
            break

    print('Bye!')


