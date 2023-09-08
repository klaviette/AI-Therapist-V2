import numpy as np
import pandas as pd
import os
import cv2

b1 = np.load('AI Therapist/Hidden Layer/Biases1.npy')
b2 = np.load('AI Therapist/Hidden Layer/Biases2.npy')
W1 = np.load('AI Therapist/Hidden Layer/Weights1.npy')
W2 = np.load('AI Therapist/Hidden Layer/Weights2.npy')

DATADIR = "AI Therapist/"
CATEGORIES = ["testing"]

data = []
IMG_SIZE = 50

def create_testing_data():

    path = os.path.join(DATADIR, CATEGORIES[0])

    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        flattened_pixels = new_array.flatten().tolist()

        data.append(flattened_pixels)
        
create_testing_data()

def get_predictions(A2):
    return np.argmax(A2, 0)

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return A2

def make_predictions(X, W1, b1, W2, b2):
    A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

data = np.array(data)

data_t = data[0:1].T
X_t = data_t

if make_predictions(X_t, W1, b1, W2, b2) == 1:
    print("Sad")
else:
    print("Happy")