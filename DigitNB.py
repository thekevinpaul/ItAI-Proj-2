import numpy as np
import matplotlib as plt
import time

def get_label(Labels):
    file_lines = open(Labels).readlines()
    file_lines=  [int(line.strip()) for line in file_lines]
    for i in range(len(file_lines)):
        if file_lines[i]<=0:
            file_lines[i]=0
    return file_lines,len(file_lines)

def load_file_images(filename,l):
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
    return Images

def proccessingData(FileData,FileLabel):
    file_lines,lenLabel= get_label(FileLabel)
    image_data=load_file_images(FileData,lenLabel)
    FlattenData=[]
    for i in range(len(image_data)):
        FlattenData.append(image_data[i].flatten())
    S_data=np.random.shuffle(np.arange(int(len(FlattenData))))
    return np.squeeze(np.array(FlattenData)[S_data]),np.squeeze(np.array(file_lines)[S_data])

def probability(x_train,y_train,f):
    train_len = x_train.shape[0]
    length=y_train.shape[0]
    Num_of_Labels=np.unique(y_train).shape[0]
    Num_of_Feature=x_train.shape[1]
    Features = f*f +1
    count_fL=np.zeros((Num_of_Labels,Num_of_Feature,Features))
    count_L=[0]*Num_of_Labels

    for i in range(train_len):
        label_name = int(y_train[i])
        count_L[label_name]+=1
        for j in range(Num_of_Feature):
            f_val=int(x_train[i,j])
            count_fL[label_name,j,f_val] = count_fL[label_name,j,f_val]+1
    prob_fl=np.zeros_like(count_fL)
    prior = np.zeros(Num_of_Labels)
    for i in range(Num_of_Labels):
        prob_fl[i,:,:]=count_fL[i,:,:]/count_L[i]
        prior[i]=count_L[i]/length
    return prob_fl,prior
def buildModel(data,probabilityFL,prior):
    NumofLabels=probabilityFL.shape[0]
    length=data.shape[0]
    NumOfFeatures=data.shape[1]
    p=np.ones((NumofLabels,length))
    pred_y=np.zeros(length)
    for i in range(NumofLabels):
        for j in range(length):
            for k in range(NumOfFeatures):
                val = int(data[j,k])
                if(probabilityFL[i,k,val]<=0.01):
                    probabilityFL[i,k,val]=0.01
                p[i,j]=p[i,j]*probabilityFL[i,k,val]
            p[i,j] = p[i,j]*prior[i]
    for i in range(length):
        pred_y[i]=np.argmax(p[:,i])
    return pred_y

def Accuracy(pred_y,true_y):
    l=pred_y.shape[0]
    c=0
    for i in range(l):
        if(pred_y[i] == true_y[i]):
            c=c+1
            print(pred_y[i], true_y[i])
    Accuracy=c/l
    return Accuracy

def main():
    trainData="data/digit/trainingimages"
    DataLabels="data/digit/traininglabels"
    testData ="data/digit/testimages"
    TestLabels="data/digit/testlabels"
    x_train,y_train=proccessingData(trainData,DataLabels)
    x_test, y_test=proccessingData(testData,TestLabels)
    DataPercent=int(x_train.shape[0]/10)
    timeTaken=[]
    TestAccuracy=[]

    for i in range(10):
        s=time.time()
        probabilityFL,prior=probability(x_train[0:DataPercent*(i+1)],y_train[0:DataPercent*(i+1)],1)
        end=time.time()
        timeTaken.append(end-s)
        pred_y = buildModel(x_test,probabilityFL,prior)
        TestAccuracy.append(Accuracy(pred_y,y_test))
        print("Training Data percent = ", (i + 1)*10, " Time taken = ", timeTaken[i], " Accuracy = ", TestAccuracy[i])
    # for i in range(10):
    #     print("Data percent = ", (i+1)*10, " Time taken = ",timeTaken[i]," Accuracy = ",TestAccuracy[i])

main()
