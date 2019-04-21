import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import fashion_mnist
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras_tqdm import TQDMNotebookCallback

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train, y_train = x_train.reshape(len(x_train), 28*28) / 255., to_categorical(y_train)
x_test, y_test = x_test.reshape(len(x_test), 28*28) / 255., to_categorical(y_test)

print("Training Dataset Size: ", x_train.shape)
print("Training Labels Size: ", y_train.shape)
print("Testing Dataset Size: ", x_test.shape)
print("Testing Labels Size: ", y_test.shape)

def display_digits(X, digit_size=28, n=10):
    figure = np.zeros((digit_size * n, digit_size * n))
    
    for i in range(n):
        for j in range(n):
            index = np.random.randint(0, X.shape[0])
            digit = X[index].reshape(digit_size, digit_size)
            
            x = i * digit_size
            y = j * digit_size
            figure[x:x + digit_size, y:y + digit_size] = digit
    
    plt.figure(figsize=(n, n))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()
    
    
display_digits(x_train)

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float 
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])


def mymodel():
    
    model = tf.keras.Sequential()
    # Define the input shape in the first layer of the neural network
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu',kernel_initializer='he_normal', input_shape=(28,28,1))) 
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.25))
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Conv2D(128, kernel_size=3, activation='relu'))
    #model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dropout(0.4))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    return model

 
batch_size = 128
epochs = 50

# Test function
mymodel().summary()

def evaluate_model(runs=5):

    
    scores = [] 
    for i in range(runs):
        print('Executing run %d' % (i+1))
        model = mymodel()
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, 
                            callbacks=[TQDMNotebookCallback()], verbose=0)
        scores.append(model.evaluate(x_test, y_test, verbose=0))
        print(' * Test set Loss: %.4f, Accuracy: %.4f' % (scores[-1][0], scores[-1][1]))
        
    accuracies = [score[1] for score in scores]     
    return np.mean(accuracies), np.std(accuracies)


 mean_accuracy, std_accuracy = evaluate_model(runs=5)

#Mean test set accuracy over 5 runs
print('Mean test set accuracy over 5 runs: %.4f +/- %.4f' % (mean_accuracy, std_accuracy))
