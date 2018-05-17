from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
model = Sequential()

model.add(Dense(units=64, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

x_test = y_test = 0

for epoch in range(1500):
    x_train, y_train = mnist.train.next_batch(100)

    model.train_on_batch(x_train, y_train)

    if epoch%100 == 0:
        print('Epoch:', str(epoch))
        x_test = mnist.test.images
        y_test = mnist.test.labels
        loss, acc = model.evaluate(x_test, y_test, batch_size=100)
        print("Loss: ", str(loss), "\nAcc: ", str(acc))
        