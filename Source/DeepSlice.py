from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

seed = 9
numpy.random.seed(seed)

# load datasets
#csv files were filtered based on the data.
input_file = "filtereddata.txt"
# test_file = "C:\\XXX.csv"

dataset = pd.read_csv(input_file).values

# read training data
# datasetTest = pd.read_csv(test_file).values

# split into input (X) and output (Y) variables
X = dataset[:,0:8].astype("int32")
Y = dataset[:,8]
# XT = datasetTest[:,0:8].astype("int32")

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = to_categorical(encoded_Y)

(X_train, X_test, Y_train, Y_test) = train_test_split(X, dummy_y, test_size=0.001, random_state=seed)
# create model
model = Sequential()
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(3, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

# Fit the model

history = model.fit(X_train, Y_train, validation_split=0.2, epochs=16)

# evaluate the model
scores = model.evaluate(X_test, Y_test)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# plot_model(model, to_file='./model.png')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper right')
plt.show()

#from sklearn.metrics import confusion_matrix
#y_pred_keras = model.predict_classes(XT)

#csv = open("C:\\DeepSlice\\5G\\output.csv", "w")
#"w" indicates that you're writing strings to the file

#pd.DataFrame(y_pred_keras).to_csv("C:\\DeepSlice\\5G\\output.csv")
#cm = confusion_matrix(Y_test, y_pred_keras, labels=[0, 1, 2])

#csv = open("C:\\DeepSlice\\5G\\input.csv", "w")
#"w" indicates that you're writing strings to the file

#pd.DataFrame(XT).to_csv("C:\\DeepSlice\\5G\\input.csv")