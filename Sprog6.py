import numpy as np
import csv
def read_data(filename):
    with open(filename,'r') as csvfile:
        datareader = csv.reader(csvfile)
        metadata = next(datareader)
        traindata=[]
        for row in datareader:
            traindata.append(row)
        print(traindata,"\n")
        print("Metadata is:\n",metadata)
    return (metadata,traindata)

def splitDataset(dataset,splitratio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet = []
    testset = list(dataset)
    i=0
    while len(trainSet)<trainSize:
        trainSet.append(testset.pop(i))
    return [trainSet,testset]

def classify(train,test):
    train_rows = train.shape[0]
    test_rows=test.shape[0]
    train_col=train.shape[1]
    test_col=test.shape[1]
    print("training data size=",train_rows)
    print("test data size=",test.shape[0])
    countYes,countNo,probYes,probNo=0,0,0,0  
    print("target  count   probability")
    for x in range(train_rows):
        if train[x,train_col-1] == 'yes':
            countYes +=1
        if train[x,train_col-1] == 'no':
            countNo +=1
    probYes=countYes/train_rows
    probNo=countNo/train_rows
    print('Yes',"\t",countYes,"\t",probYes)
    print('No',"\t",countNo,"\t",probNo)
    prob0=np.zeros((test_col-1))
    prob1=np.zeros((test_col-1))
    accuracy=0
    for t in range (test_rows):
        for k in range (test.shape[1]-1):
            count1,count0=0,0
            for j in range (train_rows):
                if test[t,k] == train[j,k] and train[j,train_col-1]=='no':
                    count0+=1
                if test[t,k]==train[j,k] and train[j,train_col-1]=='yes':
                    count1+=1
            prob0[k]=count0/countNo
            prob1[k]=count1/countYes
        probno=probNo
        probyes=probYes
        for i in range(test_col-1):
          probno=probno*prob0[i]
          probyes=probyes*prob1[i]
          if probno>probyes:
              predict='no'
          else:
              predict='yes'
          if predict == test[t,test_col-1]:
              accuracy+=1
        final_accuracy=(accuracy/test_rows)*50
        print("accuracy",final_accuracy,"%")
        return
metadata,traindata=read_data("C:\\Users\\CSE-07\\.spyder-py3\\Spro6.csv")
splitRatio=0.8
trainingset,testset=splitDataset(traindata,splitRatio)
training=np.array(trainingset)
testing=np.array(testset)
print("Training :\n",training,"\nTesting: \n",testing)
classify(training,testing)                    