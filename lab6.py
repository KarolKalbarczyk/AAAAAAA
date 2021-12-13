from tensorflow.keras.datasets import mnist
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def fullyConnected(xTrain, yTrain, xTest, yTest):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=0.005)
    model = tf.keras.models.Sequential([ 
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_uniform'),
     tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform'),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
     tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    r = model.fit(xTrain, yTrain, epochs=15, callbacks=[callback])
    epochs = len(r.history['loss'])
    test_loss, test_acc = model.evaluate(xTest,  yTest, verbose=2)
    return test_loss, test_acc, epochs


def convolutional(xTrain, yTrain, xTest, yTest):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=0.005)
    model = tf.keras.models.Sequential([ 
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
     tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    r = model.fit(xTrain, yTrain, epochs=15, callbacks=[callback])
    epochs = len(r.history['loss'])
    test_loss, test_acc = model.evaluate(xTest,  yTest, verbose=2)
    return test_loss, test_acc, epochs



def pooling(xTrain, yTrain, xTest, yTest):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=0.005)
    model = tf.keras.models.Sequential([ 
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
     tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    r = model.fit(xTrain, yTrain, epochs=15, callbacks=[callback])
    epochs = len(r.history['loss'])
    test_loss, test_acc = model.evaluate(xTest,  yTest, verbose=2)
    return test_loss, test_acc, epochs



def dropout(xTrain, yTrain, xTest, yTest):
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1, min_delta=0.005)
    model = tf.keras.models.Sequential([ 
     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
     tf.keras.layers.Flatten(input_shape=(28, 28)),
     tf.keras.layers.Dropout(0.5),
     tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform'),
     tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    r = model.fit(xTrain, yTrain, epochs=15, callbacks=[callback])
    epochs = len(r.history['loss'])
    test_loss, test_acc = model.evaluate(xTest,  yTest, verbose=2)
    return test_loss, test_acc, epochs


def calcAvg(xTrain, yTrain, xTest, yTest, fun):
    loss = 0
    acc = 0
    epochs = 0
    retries = 10
    for i in range(0, retires):
        l, a, e = fun(xTrain, yTrain, xTest, yTest)
        loss += l
        acc += a
        epochs += e
    print(loss / retries)
    print(acc / retries)
    print(epochs / retries)

if __name__ == '__main__':
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    
    xTrain = xTrain.astype("float32") / 255
    xTest = xTest.astype("float32") / 255

    xTrain = xTrain.reshape(60000, 28, 28)
    print(xTrain[0])
    xTest = xTest.reshape(10000, 28, 28)
    yTrain = to_categorical(yTrain)
    print(yTrain[0])
    yTest = to_categorical(yTest)
    
    calcAvg(xTrain, yTrain, xTest, yTest, fullyConnected)
    calcAvg(xTrain, yTrain, xTest, yTest, convolutional)
    calcAvg(xTrain, yTrain, xTest, yTest, pooling)
    calcAvg(xTrain, yTrain, xTest, yTest, dropout)
