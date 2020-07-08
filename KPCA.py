# -*- coding: utf-8 -*
import numpy as np
import os
import  matplotlib.image as mpimg
from scipy import misc
from skimage.transform import resize
from scipy.spatial.distance import pdist,squareform
from sklearn.metrics.pairwise import euclidean_distances
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
def subvector(target_matrix, target_vector):
    vector4matrix = np.repeat(target_vector, target_matrix.shape[0],axis = 0)
    target_matrix = target_matrix - vector4matrix
    return target_matrix
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

def selectkernel(kernel,X,para):
    if kernel=='linear':
        K=np.dot(X,X.T)
    elif kernel=='poly':
        k=np.dot(X,X.T)+1
        K=k**para
    elif kernel=='rbf':
        dists=pdist(X)**2
        mat=squareform(dists)
        K=np.exp(-mat/(2*(para**2)))
    return K
def kernel_Newdata(kernel,Y,X,para):
    if kernel=='linear':
        K=np.dot(Y,X.T)
    elif kernel=='poly':
        k=np.dot(Y,X.T)+1
        K=k**para
    elif kernel=='rbf':
        dists=euclidean_distances(Y,X)**2
        K=np.exp(-dists/(2*(para**2)))
    return K
    

# main program
train_face,train_face_number,test_face,test_face_number = loadimage(os.getcwd()+'/att_faces')
print(train_face.shape,test_face.shape)
para=pdist(train_face)**2
print(para.shape)
para=np.sqrt(squareform(para))
para[para<=0]=float("inf")
para=np.min(para,axis=0)
para=5*np.mean(para)
print('para',para)
#select the principal components and map face images into low dimensional space
K=selectkernel('rbf',train_face,para)
N=K.shape[0]
one_n=np.ones([N,N])/N
K_hat=K-one_n.dot(K)-K.dot(one_n)+one_n.dot(K).dot(one_n)
eigvals,eigvecs=np.linalg.eigh(K_hat)
arg_max=eigvals.argsort()[-int(eigvecs.shape[1]*principal_percent):]
eigvecs_maxnormal=eigvecs[:,arg_max]/np.sqrt(eigvals[arg_max])

#select the principal components and map face images into low dimensional space
K_test=kernel_Newdata('rbf',test_face,train_face,para)
#print(K_test.shape)
#L=K_test.shape[0]
#two_n=np.ones([L,N])/N
#K_test_hat=K_test-K_test.dot(one_n)-two_n.dot(K)+two_n.dot(K).dot(one_n)

train_f=np.dot(K,eigvecs_maxnormal)
#test_f=np.dot(K_test_hat,eigvecs_maxnormal)
test_f=np.dot(K_test,eigvecs_maxnormal)
print('train,test',train_f.shape,test_f.shape)


# recognise via measuring educlidean distance in high dimentional space
count = 0
for i in np.linspace(0, test_face.shape[0] - 1, test_face.shape[0]).astype(np.int64):
    sub = train_f-test_f[i, :]
    dis = np.linalg.norm(sub, axis = 1)#求二范数
    fig = np.argmin(dis)
    if train_face_number[fig] == test_face_number[i]:
        count = count + 1
correct_rate = count / test_face.shape[0]

# show the parameters and results
print("Principal rate=", principal_percent * 100, "%, count for", int(eigvecs.shape[1]*principal_percent), "principal eigenvectors")
print("Correct rate =", correct_rate * 100 , "%")
