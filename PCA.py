# -*- coding: utf-8 -*
import numpy as np
import os
import  matplotlib.image as mpimg
from scipy import misc
from skimage.transform import resize
import argparse
from PIL import Image
import matplotlib.pyplot as plt
# initialize parameters
parser = argparse.ArgumentParser(description='PCA ORL')
parser.add_argument('--image-scale', type=float, default=0.5, metavar='scale',
                    help='scale rate for image (default: 0.5)')
parser.add_argument('--train-per-person', type=int, default=6, metavar='k',
                    help='training number per-person minimal to 1, maximum to 9 (default: 4)')
parser.add_argument('--print-feature-face', type=bool, default=False, metavar='feature_face',
                    help='print feature face (default: False)')
parser.add_argument('--principal-rate', type=float, default=1, metavar='principal_percent',
                    help='random seed (default: 1)')
args = parser.parse_args()

scale = args.image_scale
k = args.train_per_person
feature_face = args.print_feature_face
principal_percent = args.principal_rate

# covert image to sole vector
def img2vector(filename):
    a=mpimg.imread(filename)
    imgVector =resize(a,(int(112 * scale),int(92 * scale))).flatten()
    return imgVector

# load image from diretion
def loadimage(dataSetDir):
    train_face = np.zeros((40 * k, int(112 * scale) * int(92 * scale)))  # image size:112*92
    train_face_number = np.zeros(40 * k).astype(np.int8)
    test_face = np.zeros((40 * (10 - k), int(112 * scale) * int(92 * scale)))
    test_face_number = np.zeros(40 * (10 - k)).astype(np.int8)
    for i in np.linspace(1, 40, 40).astype(np.int8): #40 sample people
            people_num = i
            for j in np.linspace(1, 10, 10).astype(np.int8): #everyone has 10 different face
                if j <= k:
                    filename = dataSetDir+'/s'+str(people_num)+'/'+str(j)+'.pgm'
                    img = img2vector(filename)
                    train_face[(i-1)*k+(j-1),:] = img
                    train_face_number[(i-1)*k+(j-1)] = people_num
                else:
                    filename = dataSetDir+'/s'+str(people_num)+'/'+str(j)+'.pgm'
                    img = img2vector(filename)
                    test_face[(i-1)*(10-k)+(j-k)-1,:] = img
                    test_face_number[(i-1)*(10-k)+(j-k)-1] = people_num

    return train_face,train_face_number,test_face,test_face_number #tuple

# subtract a vector from a matrex
def subvector(target_matrix, target_vector):
    vector4matrix = np.repeat(target_vector, target_matrix.shape[0],axis = 0)
    target_matrix = target_matrix - vector4matrix
    return target_matrix

# both data subtract mean data of train data
def submean(train_data, test_data):
    mean_data = train_data.mean(axis = 0).reshape(1, train_data.shape[1])
    train_data = subvector(train_data, mean_data)
    test_data = subvector(test_data, mean_data)
    return mean_data,train_data,test_data

# main program
train_face,train_face_number,test_face,test_face_number = loadimage(os.getcwd()+'/att_faces')
mean_data,train_face, test_face = submean(train_face, test_face)
# math calculate: the relevant mathematical formula can be found in the doc document
cov = np.dot(train_face.T, train_face)
# calculate eigenvalues & eigenvectors
l, v = np.linalg.eig(cov)
# sort eigenvectors by the value of eigenvalues
index=np.argsort(l)[::-1]
v=v[:,index]
#compute contribution rate
##ratio=0
##K=0
##for k in range(l.shape[1]):
##    r=l[0]/np.sum(l)
##    ratio=ratio+r
##    if ratio>=0.9:
##        K=k
##        break
# select the principal components and map face images into low dimensional space
v = v[:,0:int(v.shape[1]*principal_percent)]
train_face = np.dot(train_face, v)
test_face = np.dot(test_face , v)
#show feature maps
if feature_face == True:
    plt.figure('Feature Maps')
    r, c = (4, 10)
    for i in range(r * c):
        plt.subplot(r,c,i+1)
        plt.imshow(v[:, i].real.reshape(int(112 * scale), int(92 * scale)), cmap='gray')
        plt.axis('off')
    plt.show()

#利用低维度数据来重构训练集人脸
recons=np.dot(train_face,v.T)+mean_data
plt.figure('reconstrcut Maps')
r, c = (4, 10)
for i in range(r * c):
    plt.subplot(r,c,i+1)
    plt.imshow(recons[i, :].real.reshape(int(112 * scale), int(92 * scale)), cmap='gray')
    plt.axis('off')
plt.show()
# recognise via measuring educlidean distance in high dimentional space
count = 0
for i in np.linspace(0, test_face.shape[0] - 1, test_face.shape[0]).astype(np.int64):
    sub = subvector(train_face,test_face[i, :].reshape((1,test_face.shape[1])))
    dis = np.linalg.norm(sub, axis = 1)#求二范数
    fig = np.argmin(dis)
    if train_face_number[fig] == test_face_number[i]:
        count = count + 1
correct_rate = count / test_face.shape[0]

# show the parameters and results
print("Principal rate=", principal_percent * 100, "%, count for", int(v.shape[1]*principal_percent), "principal eigenvectors")
print("Correct rate =", correct_rate * 100 , "%")
