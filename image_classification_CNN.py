'''
It seems that classification into 10 categories can be interpretted as a series of 
binary classification problems which would imply that classification across 10 categories
is more difficult than binary classification. For example, if there were only 3 categories: a, b, c,
then the problem could be interpretted as a combination of (a or everything else) and (b or everything else)
and (c or everything else). As there are more and more categories, the combinations of binary classifiers
would become more complex. Therefore, binary classifcation is easier than classifying across many categories. 

'''


import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import optimizers

def load_cifar10():
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    ytrain_1hot = np.zeros(shape=(len(ytrain), 10))

    for x in range(len(ytrain)):
        temp = ytrain[x][0];
        ytrain_1hot[x][temp-1] = 1


    ytest_1hot = np.zeros(shape=(len(ytest), 10))

    for x in range(len(ytest)):
        temp = ytest[x][0];
        ytest_1hot[x][temp-1] = 1
 
    xtrain = xtrain / 255
    xtest = xtest / 255

    return xtrain, ytrain_1hot, xtest, ytest_1hot


def build_multilayer_nn():
    #10000/10000 [==============================] - 1s 83us/step
    #[1.4302016170501708, 0.50109999999999999]    

    nn = Sequential()
    nn.add(Flatten(input_shape=(32,32,3)))
    hidden = Dense(units=100, activation="relu")
    nn.add(hidden)

    output = Dense(units=10, activation="softmax")
    nn.add(output)
    return nn

def train_multilayer_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain_1hot, epochs=30, batch_size=32)        
 

def build_convolution_nn():
    #10000/10000 [==============================] - 18s 2ms/step
    #[0.7925042769432068, 0.72289999999999999]     

    nn = Sequential()
    conv1 = Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(32,32,3))
    conv2 = Conv2D(32, (3,3), activation='relu', padding="same")
    nn.add(conv1)
    nn.add(conv2)

    pool1 = MaxPooling2D(pool_size=(2,2))
    nn.add(pool1)

    drop1 = Dropout(0.25)
    nn.add(drop1)

    conv3 = Conv2D(32, (3,3), activation='relu', padding="same")
    conv4 = Conv2D(32, (3,3), activation='relu', padding="same")
    nn.add(conv3)
    nn.add(conv4)

    pool2 = MaxPooling2D(pool_size=(2,2))
    nn.add(pool2)

    drop2 = Dropout(0.25)
    nn.add(drop2)

    nn.add(Flatten())

    hidden1 = Dense(units=250, activation="relu")
    nn.add(hidden1)

    hidden2 = Dense(units=600, activation="relu")
    nn.add(hidden2)

    output = Dense(units=10, activation="softmax")
    nn.add(output)
    return nn
    

def train_convolution_nn(model, xtrain, ytrain_1hot):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain_1hot, epochs=20, batch_size=32)        
     

def get_binary_cifar10():   
    # 1 indicates animal and 0 indicates vehicle 
    train, test = cifar10.load_data()
    xtrain, ytrain = train
    xtest, ytest = test

    ytrain_binary = np.zeros(shape=(len(ytrain),1))

    for x in range(len(ytrain)):
        temp = ytrain[x][0];
        if temp>=2 and temp<=7:
            ytrain_binary[x] = [1]

    ytest_binary = np.zeros(shape=(len(ytest),1))

    for x in range(len(ytest)):
        temp = ytest[x][0];
        if temp>=2 and temp<=7:
            ytest_binary[x] = [1]


    xtrain = xtrain / 255
    xtest = xtest / 255

    return xtrain, ytrain_binary, xtest, ytest_binary


def build_binary_classifier(): 
    #10000/10000 [==============================] - 18s 2ms/step
    #[0.17787481148242951, 0.92800000000000005]

    nn = Sequential()
    conv1 = Conv2D(32, (3,3), activation='relu', padding="same", input_shape=(32,32,3))
    conv2 = Conv2D(32, (3,3), activation='relu', padding="same")
    nn.add(conv1)
    nn.add(conv2)

    pool1 = MaxPooling2D(pool_size=(2,2))
    nn.add(pool1)

    drop1 = Dropout(0.25)
    nn.add(drop1)

    conv3 = Conv2D(32, (3,3), activation='relu', padding="same")
    conv4 = Conv2D(32, (3,3), activation='relu', padding="same")
    nn.add(conv3)
    nn.add(conv4)

    pool2 = MaxPooling2D(pool_size=(2,2))
    nn.add(pool2)

    drop2 = Dropout(0.25)
    nn.add(drop2)

    nn.add(Flatten())

    hidden1 = Dense(units=250, activation="relu")
    nn.add(hidden1)

    hidden2 = Dense(units=600, activation="relu")
    nn.add(hidden2)

    output = Dense(units=1, activation="sigmoid")
    nn.add(output)
    return nn


def train_binary_classifier(model, xtrain, ytrain_binary):
    sgd = optimizers.SGD(lr=0.01)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 
    model.fit(xtrain, ytrain_binary, epochs=20, batch_size=32)  


if __name__ == "__main__":

    
    xtrain, ytrain_binary, xtest, ytest_binary = get_binary_cifar10()
    nn = build_binary_classifier()
    train_binary_classifier(nn, xtrain, ytrain_binary)

    print(nn.evaluate(xtest, ytest_binary))
    

    '''
    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()
    nn = build_multilayer_nn()
    train_multilayer_nn(nn, xtrain, ytrain_1hot)

    print(nn.evaluate(xtest, ytest_1hot))
    '''
    '''
    xtrain, ytrain_1hot, xtest, ytest_1hot = load_cifar10()
    nn = build_convolution_nn()
    train_convolution_nn(nn, xtrain, ytrain_1hot)

    print(nn.evaluate(xtest, ytest_1hot))
'''
    # Write any code for testing and evaluation in this main section.


