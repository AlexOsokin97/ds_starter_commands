# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:05:20 2020

@author: Alexander
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Loading the Dataset 
data = pd.read_csv('ML_df.csv')

#creating a copy of the data set
df = data.copy()

#Dropping the irrelevant columns
df.drop(['Unnamed: 0', 'App', 'Last Updated', 'numeric_date', 'Genres'], axis=1,inplace=True)

#creating X, y
X = df.iloc[:, [0,2,3,4,5,6,8]]
y = df.iloc[:, 1]

#Encoding the categorical data with dummy variables
X = pd.get_dummies(X, columns=["Category", "Content Rating"], prefix=["catgry", "cr"])
X.drop(labels=["catgry_ART_AND_DESIGN","cr_Everyone"], axis=1, inplace=True)
X = X.values
y = y.values

#splitting the data into training and testing 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

#applying feature scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:, 0:5] = scaler.fit_transform(X_train[:, 0:5])
X_test[:, 0:5] = scaler.transform(X_test[:, 0:5])

#importing keras libraries and packages for the NN model
import keras
from keras.models import Sequential 
from keras.layers import Dense

#initializing the ANN
neural = Sequential()

#creating NN input layer
neural.add(Dense(activation="relu", input_dim=39, units=20, kernel_initializer="he_uniform"))

#creating the NN hidden layers
neural.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
neural.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
neural.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
neural.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))

#creating NN output layer
neural.add(Dense(activation="linear", input_dim=20, units=1, kernel_initializer="he_uniform"))

#compiling the ANN
neural.compile(optimizer = "adagrad" , loss="mean_squared_logarithmic_error" , metrics=['mse'])

#fitting the model
history = neural.fit(X_train, y_train, validation_data=(X_test,y_test), batch_size=10, epochs=100, verbose=0)

# evaluate the model
train_mse = neural.evaluate(X_train, y_train, verbose=0)
test_mse = neural.evaluate(X_test, y_test, verbose=0)

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot mse during training
plt.subplot(212)
plt.title('Mean Squared Error')
plt.plot(history.history['mse'], label='train')
plt.plot(history.history['val_mse'], label='test')
plt.legend()
plt.show()


###############################################################################################

#initializing the ANN
model = Sequential()

#creating NN input layer
model.add(Dense(activation="relu", input_dim=39, units=20, kernel_initializer="he_uniform"))
          
#creating the NN hidden layers
model.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
model.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))

#creating NN output layer          
model.add(Dense(activation="linear", input_dim=20, units=1, kernel_initializer="he_uniform"))

#compiling the ANN
model.compile(optimizer = "sgd" , loss="mean_squared_error", metrics=["mse"])

# fit model
history2 = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

# evaluate the model
train_mse_model = model.evaluate(X_train, y_train, verbose=0)
test_mse_model = model.evaluate(X_test, y_test, verbose=0)

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history2.history['loss'], label='train')
plt.plot(history2.history['val_loss'], label='test')
plt.legend()
# plot mse during training
plt.subplot(212)
plt.title('Mean Squared Error')
plt.plot(history2.history['mse'], label='train')
plt.plot(history2.history['val_mse'], label='test')
plt.legend()
plt.show()

###############################################################################################

#initializing the ANN
model2 = Sequential()

#creating NN input layer
model2.add(Dense(activation="relu", input_dim=39, units=20, kernel_initializer="he_uniform"))
          
#creating the NN hidden layers
model2.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
model2.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
model2.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
model2.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
model2.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))
model2.add(Dense(activation="relu" , input_dim=20, units=20, kernel_initializer="he_uniform"))

#creating NN output layer          
model2.add(Dense(activation="linear", input_dim=20, units=1, kernel_initializer="he_uniform"))

#compiling the ANN
model2.compile(optimizer = "sgd" , loss="mean_absolute_error", metrics=['mse'])

# fit model
history3 = model2.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, verbose=0)

# evaluate the model
train_mse_model2 = model2.evaluate(X_train, y_train, verbose=0)
test_mse_model2 = model2.evaluate(X_test, y_test, verbose=0)

# plot loss during training
plt.subplot(211)
plt.title('Loss')
plt.plot(history3.history['loss'], label='train')
plt.plot(history3.history['val_loss'], label='test')
plt.legend()
# plot mse during training
plt.subplot(212)
plt.title('Mean Squared Error')
plt.plot(history3.history['mse'], label='train')
plt.plot(history3.history['val_mse'], label='test')
plt.legend()
plt.show()

##############################################################################################
neural.save("neural_network.h5")
print("Saved model to disk")





