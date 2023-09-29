# AI-Therapist-V2

Description:
This program provides an overview of a neural network created with Numpy (and without TensorFlow).
It takes in a 2D image of a face (either happy or sad) and will display its prediction
as to what the face is doing (smiling/frowning).

Originally, this was created on Kaggle and some of the files may be .ipynb. These notebook files
are not important to the actual program, the weights and biases after training the model were exported to .npy
files and can be used separately, as the below section explains.

How to use:
The file that will run is main.py; and weights and biases in the form of .npy files are required:

"""
b1 = np.load('AI Therapist/Hidden Layer/Biases1.npy')
b2 = np.load('AI Therapist/Hidden Layer/Biases2.npy')
W1 = np.load('AI Therapist/Hidden Layer/Weights1.npy')
W2 = np.load('AI Therapist/Hidden Layer/Weights2.npy')
"""

you can store them on your local directory and make sure to reference them in the program.
An image is also required, any size will work as the program automatically resizes it in pre-
processing. 
