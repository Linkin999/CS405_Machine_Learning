import numpy as np

class LogisticRegression():
    def __init__(self,n_features,n_classes,max_epoch,lr) -> None:
        self.w=np.zeros((n_features+1,n_classes))
        self.max_epoch=max_epoch
        self.lr=lr

    def softMaxLayer(self,x):
        exp_x=np.exp(x-np.max(x,axis=1,keepdims=True))
        partition=exp_x/exp_x.sum(axis=1,keepdims=True)
        return partition

    def forward_propagation(self,X):
        Z=np.dot(X,self.w)
        y_hat=self.softMaxLayer(Z)
        return y_hat
    
    def compute_loss(self,y_hat,Y):
        m=Y.shape[0]
        loss=-np.sum(Y*np.log(y_hat))/m
        return loss
    
    def backward_propagation(self,X,y_hat,Y):
        m=X.shape[0]
        dZ=y_hat-Y
        dW=np.dot(X.T,dZ)/m
        return dW
    
    def update_parameters(self,dw):
        self.w=self.w-self.lr*dw
        return self.w

    def fit(self,X,y):
        X=np.concatenate([X,np.ones((X.shape[0],1))],axis=-1)
        
        #初始化W
        #self.w=np.random.random((self.w.shape[0],self.w.shape[1]))

        for epoch in range(0,self.max_epoch):
            y_hat=self.forward_propagation(X)
            loss=self.compute_loss(y_hat,y)
            #print('epoch %d, loss %.4f'% (epoch + 1, loss))
            dw=self.backward_propagation(X,y_hat,y)
            self.w=self.update_parameters(dw)
        
        output=self.w.reshape(-1)
        for i in range(len(output)):
            print('%.3f'% output[i])

    def prerict(self,X_test):
        X_test=np.concatenate([X_test,np.ones((X_test.shape[0],1))],axis=-1)
        Z=self.forward_propagation(X_test)
        return np.argmax(Z,axis=1)
    
    # def main():
    #     firstLine=input().split(' ')
    #     X_train=[]
    #     Y_train=[]

    #     for i in range(0,int(firstLine[0])):
    #         inputs=input().split(' ')
    #         # print(inputs)
    #         X_train.append([float(feature) for feature in inputs[0:4]])
    
    #     for i in range(0,int(firstLine[0])):
    #         inputs=input().split(' ')
    #         Y_train.append([int(feature) for feature in inputs[0:3]])
    
    #     X_train = np.array(X_train)
    #     Y_train = np.array(Y_train)

    #     model=LogisticRegression(n_features=int(firstLine[1]),n_classes=int(firstLine[2]),max_epoch=int(firstLine[3]),lr=float(firstLine[4]))
    #     model.fit(X_train,Y_train)

    #     output=model.w.reshape(-1)
    #     for i in range(len(output)):
    #         print('%.3f'% output[i])

if __name__ == '__main__':
    firstLine=input().split(' ')
    X_train=[]
    Y_train=[]

    for i in range(0,int(firstLine[0])):
        inputs=input().split(' ')
        # print(inputs)
        # print(len(inputs))
        X_train.append([float(feature) for feature in inputs[0:int(firstLine[1])]])
    
    for i in range(0,int(firstLine[0])):
        inputs=input().split(' ')
        Y_train.append([int(feature) for feature in inputs[0:int(firstLine[2])]])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    model=LogisticRegression(n_features=int(firstLine[1]),n_classes=int(firstLine[2]),max_epoch=int(firstLine[3]),lr=float(firstLine[4]))
    model.fit(X_train,Y_train)

    # output=model.w.reshape(-1)
    # for i in range(len(output)):
    #     print('%.3f'% output[i])

    

        