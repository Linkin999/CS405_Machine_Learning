import numpy as np



def euc_distance(instance1,instance2):
    distance=np.sqrt(np.sum((instance1-instance2)**2))
    return distance

def distancesNeighbors(n_neighbors,X_train,y_train, sample):
    distance=[]
    for i in range(len(X_train)):
        distance.append([euc_distance(X_train[i],sample),y_train[i]])
    neighbors=sorted(distance,key=lambda x:x[0])[:n_neighbors]
    neighbors=np.array([i[1] for i in neighbors])
    return neighbors

def major(neighbors):
    sum0=0
    sum1=1
    for neighbor in neighbors:
        if neighbor==0:
            sum0=sum0+1
        else:
            sum1=sum1+1
    
    if(sum1>sum0):
        return 1
    else:
        return 0

def KNN(n_neighbors,X_train,y_train,X_test):
    predictions=[]
    for sample in X_test:
        neighbors = distancesNeighbors(n_neighbors,X_train,y_train,sample)
        predictions.append(major(neighbors))
    
    return np.array(predictions,dtype='float64')



if __name__ == '__main__':
    n1=int(input())
    X_train=[]
    y_train=[]

    for _ in range(n1):
        inputs=input().split(' ')
        y_train.append(int(inputs[0]))
        X_train.append([float(x) for x in inputs[1:31]])
    
    X_train=np.array(X_train)
    y_train=np.array(y_train)

    n2=int(input())
    Ks=[]
    X_test=[]
    for _ in range(n2):
        inputs=input().split(' ')
        Ks.append(int(inputs[0]))
        X_test.append([float(x) for x in inputs[1:31]])
    X_test=np.array(X_test)

    for i in range(len(X_test)):
        label=KNN(Ks[i],X_train,y_train,np.array([X_test[i]]))[0]
        print(int(label))
  
    # # fetch dataset 
    # breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
    # # data (as pandas dataframes) 
    # X = breast_cancer_wisconsin_diagnostic.data.features 
    # y = breast_cancer_wisconsin_diagnostic.data.targets 
  
    # # metadata 
    # print(breast_cancer_wisconsin_diagnostic.metadata) 
  
    # # variable information 
    # print(breast_cancer_wisconsin_diagnostic.variables) 