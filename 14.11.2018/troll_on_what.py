from scipy.misc import imread, imresize
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge#, concatenate, Reshape, Activation
from keras.models import Model, model_from_json
from keras import initializers
from keras.regularizers import l2
from keras.optimizers import SGD
from googlenet_custom_layers_new import PoolHelper, LRN
import numpy as np
from load_cifar10 import load_cifar10_data
from keras.optimizers import sgd






if __name__ == "__main__":

    # or 2. Load the full model(with pretrained weights)
    json_file = open('./googlenet/googlenet_architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("googlenet_weights.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # score = loaded_model.evaluate(X, Y, verbose=0)


    img = imresize(imread('cat.jpg', mode='RGB'), (224, 224)).astype(np.float32)
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, axis=0)

    out = loaded_model.predict(img)  # note: the model has three sets of outputs
    print(np.argmax(out[2]))

