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
import copy


rng = np.random.RandomState(1)
digits = datasets.load_digits()
images = digits.images
labels = digits.target

#Get our training data
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.6, shuffle=False)

def dataset_searcher(number_list):
  #insert code that when given a list of integers, will find the labels and images
  #and put them all in numpy arrary (at the same time, as training and testing data)

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


  #return images_nparray, labels_nparray

def print_number(images,labels):
  #insert code that when given images and labels (of numpy arrays)
  #the code will plot the images and their labels in the title. 
    unique_labels = {}
    for i, label in enumerate(labels):
        if label not in unique_labels:
            unique_labels[label] = i  # Store the index of the first occurrence

    # Extract the first instance of each label
    first_images = [images[idx] for idx in unique_labels.values()]
    first_labels = [labels[idx] for idx in unique_labels.values()]

    # Convert to numpy arrays
    first_images = np.array(first_images)
    first_labels = np.array(first_labels)

    # Number of unique labels
    num_images = len(first_labels)

    # Set up the plot grid
    cols = min(10, num_images)  # Maximum 10 columns
    rows = (num_images + cols - 1) // cols

    # Set figure size
    plt.figure(figsize=(cols * 1.5, rows * 1.5))

    # Plot each image with its label
    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(first_images[i], cmap='gray')
        plt.title(str(first_labels[i]))
        plt.axis('off')

    plt.tight_layout()
    plt.show()



class_numbers = [2,0,8,7,5]
#Part 1
class_number_images , class_number_labels = dataset_searcher(class_numbers)
#Part 2
print_number(class_number_images , class_number_labels )


model_1 = GaussianNB()

#however, before we fit the model we need to change the 8x8 image data into 1 dimension
# so instead of having the Xtrain data beign of shape 718 (718 images) by 8 by 8
# the new shape would be 718 by 64
X_train_reshaped = X_train.reshape(X_train.shape[0], -1)

#Now we can fit the model
model_1.fit(X_train_reshaped, y_train)
#Part 3 Calculate model1_results using model_1.predict()
X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
model1_results = model_1.predict(X_test_reshaped)
def OverallAccuracy(results, actual_values):
  #Calculate the overall accuracy of the model (out of the predicted labels, how many were correct?)
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


#Part 5
allnumbers = [0,1,2,3,4,5,6,7,8,9]
allnumbers_images, allnumbers_labels = dataset_searcher(allnumbers)
all_pred_gaussian_clean = model_1.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1))
print_number(allnumbers_images, all_pred_gaussian_clean)



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
  print_number(allnumbers_images, all_pred) #KNN 

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
  print_number(allnumbers_images, all_pred) #mlp

mlp()



#Part 8
#Poisoning
# Code for generating poison data. There is nothing to change here.
noise_scale = 10.0
poison = rng.normal(scale=noise_scale, size=X_train.shape)

X_train_poison = X_train + poison


#Part 9-11
#Determine the 3 models performance but with the poisoned training data X_train_poison and y_train instead of X_train and y_train

X_train_poison = X_train_poison.reshape(X_train_poison.shape[0], -1)
X_test_1 = X_test.reshape(X_test.shape[0], -1)
allnumbers_poison = allnumbers_images + rng.normal(scale=noise_scale, size=allnumbers_images.shape)

#gaussian nb
model_1_poison = GaussianNB()
model_1_poison.fit(X_train_poison, y_train)
model1_poison_results = model_1_poison.predict(X_test_1)
all_pred1 = model_1_poison.predict(allnumbers_poison.reshape(allnumbers_poison.shape[0], -1))
print_number(allnumbers_poison, all_pred1)
Model1_Poison_Accuracy = OverallAccuracy(model1_poison_results, y_test)
print("Poisoned GaussianNB accuracy:", Model1_Poison_Accuracy)

# k nearest
model_2_poison = KNeighborsClassifier(n_neighbors=10)
model_2_poison.fit(X_train_poison, y_train)

model2_poison_results = model_2_poison.predict(X_test_1)
Model2_Poison_Accuracy = OverallAccuracy(model2_poison_results, y_test)
all_pred2 = model_2_poison.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1))
print_number(allnumbers_poison, all_pred2) 
print("Poisoned KNN accuracy:", Model2_Poison_Accuracy)

#mlp
model_3_poison = MLPClassifier(random_state=0)
model_3_poison.fit(X_train_poison, y_train)

model3_poison_results = model_3_poison.predict(X_test_1)
Model3_Poison_Accuracy = OverallAccuracy(model3_poison_results, y_test)
all_pred3 = model_3_poison.predict(allnumbers_images.reshape(allnumbers_images.shape[0], -1))
print_number(allnumbers_poison, all_pred3)
print("Poisoned MLP accuracy:", Model3_Poison_Accuracy)

#Part 12-13
# Denoise the poisoned training data, X_train_poison. 
# hint --> Suggest using KernelPCA method from sklearn library, for denoising the data. 
# When fitting the KernelPCA method, the input image of size 8x8 should be reshaped into 1 dimension
# So instead of using the X_train_poison data of shape 718 (718 images) by 8 by 8, the new shape would be 718 by 64
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_poison)
kpca = KernelPCA(
    n_components=64,
    kernel="rbf",
    gamma=0.1,
    fit_inverse_transform=True,
    eigen_solver="arpack",
    random_state=0)

X_train_kpca = kpca.fit_transform(X_train_scaled)
X_denoised_scaled = kpca.inverse_transform(X_train_kpca)
 
X_train_denoised = scaler.inverse_transform(X_denoised_scaled)


#Part 14-15
#Determine the 3 models performance but with the denoised training data, X_train_denoised and y_train instead of X_train_poison and y_train
#Explain how the model performances changed after the denoising process.

allnumbers_images_scaled = scaler.transform(allnumbers_images.reshape(allnumbers_images.shape[0], -1))
allnumbers_kpca = kpca.transform(allnumbers_images_scaled)
allnumbers_denoised_scaled = kpca.inverse_transform(allnumbers_kpca)
allnumbers_denoised = scaler.inverse_transform(allnumbers_denoised_scaled).reshape(allnumbers_images.shape)

# Gaussian NB
model_1_denoise = GaussianNB()
model_1_denoise.fit(X_train_denoised, y_train)

model1_denoise_results = model_1_denoise.predict(X_test_1)
Model1_Denoise_Accuracy = OverallAccuracy(model1_denoise_results, y_test)
all_pred1_denoise = model_1_denoise.predict(allnumbers_denoised.reshape(allnumbers_denoised.shape[0], -1))
print_number(allnumbers_denoised, all_pred1_denoise)

print("Denoised GaussianNB accuracy:", Model1_Denoise_Accuracy)

# K Nearest Neighbours
model_2_denoise = KNeighborsClassifier(n_neighbors=10)
model_2_denoise.fit(X_train_denoised, y_train)

model2_denoise_results = model_2_denoise.predict(X_test_1)
Model2_Denoise_Accuracy = OverallAccuracy(model2_denoise_results, y_test)
all_pred2_denoise = model_2_denoise.predict(allnumbers_denoised.reshape(allnumbers_denoised.shape[0], -1))
print_number(allnumbers_denoised, all_pred2_denoise)
print("Denoised KNN accuracy:", Model2_Denoise_Accuracy)

# Multi-layer Perceptron
model_3_denoise = MLPClassifier(random_state=0)
model_3_denoise.fit(X_train_denoised, y_train)

model3_denoise_results = model_3_denoise.predict(X_test_1)
Model3_Denoise_Accuracy = OverallAccuracy(model3_denoise_results, y_test)
all_pred3_denoise = model_3_denoise.predict(allnumbers_denoised.reshape(allnumbers_denoised.shape[0], -1))
print_number(allnumbers_denoised, all_pred3_denoise)
print("Denoised MLP accuracy:", Model3_Denoise_Accuracy)
