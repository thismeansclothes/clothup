## model.py
import tensorflow as tf
import scipy.io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import requests
import json
import os
import pickle

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

nsamples, nx, ny = x_train.shape

d2_x_train = x_train.reshape((nsamples,nx*ny))

model = RandomForestClassifier()
#model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf = 8,
#                              min_samples_split = 20, random_state = 0, n_jobs = -1)
model.fit(d2_x_train, y_train)

f=open("model.pkl","wb")
pickle.dump(model,f)
f.close()



