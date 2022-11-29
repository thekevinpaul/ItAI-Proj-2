import numpy as np
import matplotlib as plt
import time

def sigmoid_func(s):
    return 1/(1+np.exp(-s))

def get_label(Labels):
    file_lines = open(Labels).readlines()
    file_lines=  [int(line.strip()) for line in file_lines]
    return file_lines,len(file_lines)

def load_file_images(filename,l,pool):
    file_lines=open(filename).readlines()
    file_len = int(len(file_lines))
    w= int(len(file_lines[0]))
    length= int(file_len/l)
    Images = []
    for i in range(l):
        Img = np.zeros((length,w))
        c=0
        for j in range (length*i,length*(i+1)):
            Line=file_lines[j]
            for k in range(len(Line)):
                if(Line[k]=="+" or Line[k]=="#"):
                    Img[c,k]=1
            c=c+1
        Images.append(Img)
    N_r=int(length/pool)
    N_c=int(w/pool)
    N_Images = np.zeros((l,N_r,N_c))
    for i in range(l):
        for j in range(N_r):
            for k in range(N_c):
                pix=0
                for r in range(pool*j,pool*(j+1)):
                    for c in range(pool*k,pool*(k+1)):
                        pix =pix + Images[i][r,c]
                N_Images[i,j,k]=pix
    return N_Images


def proccessingData(FileData,FileLabel,pool):
    file_lines,lenLabel= get_label(FileLabel)
    image_data=load_file_images(FileData,lenLabel,pool)
    FlattenData=[]
    for i in range(len(image_data)):
        FlattenData.append(image_data[i].flatten())
    S_data=np.random.shuffle(np.arange(int(len(FlattenData))))
    return np.squeeze(np.array(FlattenData)[S_data]),np.squeeze(np.array(file_lines)[S_data])



def trainmodel(x_train,y_train, lr,iteration):
    x = np.random.rand(x_train.shape[1],10)
    print("Y_train",y_train.shape)
    iteration_count=[]
    acc_iteration_count=[]
    for it in range(iteration):
        iteration_count.append(it+1)
        e=0
        for i in range(y_train.shape[0]):
            temp = np.zeros((len(x_train[0]), 1))
            for j in range (len(x_train[0])):
                a = x_train[i][j]
                temp[j]=a
            dot_product = np.dot(x.T,temp)
            P_temp=np.argmax(dot_product)
            if(P_temp!=y_train[i]):
                #x[:,y_train[i]] = x[:,y_train[i]] + lr*x_train[i,y_train[i]]
                e = e+1
                x[:, y_train[i]] = x[:,y_train[i]] + temp[:,0]
                x[:, P_temp] = x[:,P_temp] - temp[:,0]
        acc_iteration_count.append(100 - (e / len(x_train)) * 100)
        print(acc_iteration_count[it])
        if(e==0):
            break
    return x
def activationf(f):
    return np.where(f>0,1,0)

# def Model(x_train,y_train,lr,iteration):
#     length,n_features=x_train.shape
#     weights=np.zeros(n_features)
#     bias=0
#     y = np.where(y_train>0,1,0)
#     for i in range(iteration):
#         for index,x in enumerate(x_train):
#             f=np.dot(x,weights) + bias
#             pred_y=activationf(f)
#             update = lr*(y[index]-pred_y)
#             weights = weights+update*x
#             bias= bias+ update
#     return weights,bias

# def Prediction(weights,bias,x_test):
#     l=x_test.shape[0]
#     # temp = np.dot(x_test,x)
#     # y_pred = np.zeros(temp.shape[0])
#     # x= x.reshape(x_test.shape[1],10)
#     for i in range(l):
#         f = np.dot(x_test, weights) + bias
#         pred_y = activationf(f)
#     return pred_y

def Prediction(x,x_test):
    l=x_test.shape[0]
    temp = np.dot(x_test,x)
    y_pred = np.zeros(temp.shape[0])
    x= x.reshape(x_test.shape[1],10)
    for i in range(l):
        P_max = np.argmax(temp[i])
        y_pred[i]=P_max
    return y_pred

def Accuracy(pred_y,true_y):
    predY=np.squeeze(pred_y)
    l=predY.shape[0]
    c=0
    #print(l,true_y.shape[0])
    for i in range(l):
        if(predY[i] == true_y[i]):
            c=c+1
        print(predY[i],true_y[i])
    Accuracy=c/l
    print(c," ",l)
    return Accuracy

def main():
    trainData = "data/facedata/facedatatrain"
    DataLabels = "data/facedata/facedatatrainlabels"
    testData = "data/facedata/facedatatest"
    TestLabels = "data/facedata/facedatatestlabels"
    x_train,y_train=proccessingData(trainData,DataLabels,2)
    x_test, y_test=proccessingData(testData,TestLabels,2)
    DataPercent=int(x_train.shape[0]/10)
    timeTaken=[]
    TestAccuracy=[]

    for i in range(10):
        s=time.time()
        #probabilityFL,prior=probability(x_train[0:DataPercent*(i+1)],y_train[0:DataPercent*(i+1)],1)
        x = trainmodel(x_train[0:DataPercent*(i+1)],y_train[0:DataPercent*(i+1)],0.09,1000)
        end=time.time()
        timeTaken.append(end-s)
        pred_y = Prediction(x,x_test)
        TestAccuracy.append(Accuracy(pred_y,y_test))
        print("Training Data percent = ", (i + 1)*10, " Time taken = ", timeTaken[i], " Accuracy = ", TestAccuracy[i])
    # for i in range(10):
    #     print("Data percent = ", (i+1)*10, " Time taken = ",timeTaken[i]," Accuracy = ",TestAccuracy[i])

main()
