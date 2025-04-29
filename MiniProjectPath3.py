#MiniProjectPath3
import numpy as np
import matplotlib.pyplot as plt
# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
#import models
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.base import clone
from sklearn.preprocessing import MinMaxScaler
import copy

rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

# Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)


def dataset_searcher(number_list, images, labels):
  # insert code that when given a list of integers, will find the labels and images
  # and put them all in numpy arrary (at the same time, as training and testing data)

  wanted_classes = set(number_list)

  selected_images = []
  selected_labels = []

  for i in range(len(labels)):
    curr_label = labels[i]

    if curr_label in wanted_classes:
      selected_images.append(images[i])  # add 8×8 image
      selected_labels.append(curr_label)  # add label

  images_nparray = np.array(selected_images)
  labels_nparray = np.array(selected_labels)

  return images_nparray, labels_nparray

  #pass


def print_numbers(images, labels):
  # insert code that when given images and labels (of numpy arrays)
  # the code will plot the images and their labels in the title.

  num_images = images.shape[0]

  if num_images >= 10:
    cols = 10
  else:
    cols = num_images

  rows = (num_images + cols - 1) // cols

  #fig size
  plt.figure(figsize=(cols * 1.5, rows * 1.5))

  #loop over every sample
  for i in range(num_images):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(str(labels[i]))
    plt.axis('off')

  plt.tight_layout()
  plt.show()
  #pass


class_numbers = [2, 0, 8, 7, 5]
# Part 1
class_number_images, class_number_labels = dataset_searcher(class_numbers, images, labels)
# Part 2
#print_numbers(class_number_images, class_number_labels)

model_1 = GaussianNB()

# however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

# Now we can fit the model
model_1.fit(X_train_reshaped, y_train)

# Part 3 Calculate model1_results using model_1.predict()
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_reshaped)

def OverallAccuracy(results, actual_values):
  # Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
  correct = 0
  total = len(results)

  for i in range(total):
    if results[i] == actual_values[i]:
      correct += 1

  Accuracy = correct / total
  return Accuracy


# Part 4
Model1_Overall_Accuracy = OverallAccuracy(model1_results, y_test)
print("The overall results of the Gaussian model is " + str(Model1_Overall_Accuracy))

# Part 5
allnumbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers, images, labels)



#Part 6
#Repeat for K Nearest Neighbors
model_2 = KNeighborsClassifier(n_neighbors=10)


def knn():

  X_train_k = X_train.reshape(X_train.shape[0], -1)
  X_test_k = X_test.reshape(X_test.shape[0], -1)
  # Fit KNN
  model_2.fit(X_train_k, y_train)

  knn_results = model_2.predict(X_test_k)

  acc_knn = OverallAccuracy(knn_results, y_test)
  print("KNN model overall accuracy: " + str(acc_knn))

  all_pred = model_2.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1))
  #print_numbers(allnumbers_images, all_pred)


knn()

#Repeat for the MLP Classifier
model_3 = MLPClassifier(random_state=0)


def mlp():
  # Reshape training and test data
  X_train_m = X_train.reshape(X_train.shape[0], -1)
  X_test_m = X_test.reshape(X_test.shape[0], -1)
  # Fit MLP
  model_3.fit(X_train_m, y_train)
  mlp_results = model_3.predict(X_test_m)

  acc_mlp = OverallAccuracy(mlp_results, y_test)
  print("MLP model overall accuracy: " + str(acc_mlp))
  all_pred = model_3.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1))
 # print_numbers(allnumbers_images, all_pred)

mlp()

#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison1 = X_train + poison


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train

X_train_poison = X_train_poison1.reshape(X_train_poison1.shape[0], -1)
X_test_1 = X_test.reshape(X_test.shape[0], -1)

#gaussian nb
model_1_poison = GaussianNB()
model_1_poison.fit(X_train_poison, y_train)

model1_poison_results = model_1_poison.predict(X_test_1)
Model1_Poison_Accuracy = OverallAccuracy(model1_poison_results, y_test)
print("Poisoned GaussianNB accuracy:", Model1_Poison_Accuracy)

# k nearest
model_2_poison = KNeighborsClassifier(n_neighbors=10)
model_2_poison.fit(X_train_poison, y_train)

model2_poison_results = model_2_poison.predict(X_test_1)
Model2_Poison_Accuracy = OverallAccuracy(model2_poison_results, y_test)
print("Poisoned KNN accuracy:", Model2_Poison_Accuracy)

#mlp
model_3_poison = MLPClassifier(random_state=0)
model_3_poison.fit(X_train_poison, y_train)

model3_poison_results = model_3_poison.predict(X_test_1)
Model3_Poison_Accuracy = OverallAccuracy(model3_poison_results, y_test)
print("Poisoned MLP accuracy:", Model3_Poison_Accuracy)




#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64



#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.




kpca = KernelPCA(
    n_components=50,
    kernel="rbf",
    gamma= 0.1,
    alpha=1e-3,
    fit_inverse_transform=True,
    random_state=0,
)

Z = kpca.fit_transform(X_train_poison)
X_train_denoise = kpca.inverse_transform(Z)


# Gaussian NB
model_1_denoise = GaussianNB()
model_1_denoise.fit(X_train_denoise, y_train)

model1_denoise_results = model_1_denoise.predict(X_test_1)
Model1_Denoise_Accuracy = OverallAccuracy(model1_denoise_results, y_test)
print("Denoised GaussianNB accuracy:", Model1_Denoise_Accuracy)

# K Nearest Neighbours
model_2_denoise = KNeighborsClassifier(n_neighbors=10)
model_2_denoise.fit(X_train_denoise, y_train)

model2_denoise_results = model_2_denoise.predict(X_test_1)
Model2_Denoise_Accuracy = OverallAccuracy(model2_denoise_results, y_test)
print("Denoised KNN accuracy:", Model2_Denoise_Accuracy)

# Multi-layer Perceptron
model_3_denoise = MLPClassifier(random_state=0)
model_3_denoise.fit(X_train_denoise, y_train)

model3_denoise_results = model_3_denoise.predict(X_test_1)
Model3_Denoise_Accuracy = OverallAccuracy(model3_denoise_results, y_test)
print("Denoised MLP accuracy:", Model3_Denoise_Accuracy)
