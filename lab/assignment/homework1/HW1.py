import numpy as np
import itertools
import functools

class PolynomialFeature(object):
    def __init__(self,degree=2):
        assert isinstance(degree,int)
        self.degree=degree
    def transform(self,x):
        if x.ndim==1:
            x=x[:,None]
        x_t=x.transpose()
        features=[np.ones(len(x))]
        for degree in range(1,self.degree+1):
            for items in itertools.combinations_with_replacement(x_t,degree):
                features.append(functools.reduce(lambda x,y:x*y,items))
        return np.asarray(features).transpose()



class Regression(object):
    pass

class LinearRegression(Regression):
    def fit(self,X_train:np.ndarray,Y_train:np.ndarray):
        self.theta=np.linalg.pinv(X_train) @ Y_train
        self.var=np.mean(np.square(X_train @ self.theta-Y_train))
    
    def predict(self,X_test:np.ndarray,return_std:bool=False):
        y=X_test @ self.theta
        if return_std:
            y_std=np.sqrt(self.var)+np.zeros_like(y)
            return y,y_std
        return y

if  __name__=='__main__':
    number_training,number_testing=input('').split(' ')
    number_training,number_testing=int(number_training),int(number_testing)
    training_samples=[]#pair
    testing_samples=[]
    for i in range(number_training):
        training_pair=input('').split(' ')
        training_samples.append([float(training_pair[0]),float(training_pair[1])])
    for i in range(number_testing):
        testing_samples.append(float(input('')))

    training_samples=np.array(training_samples)
    testing_samples=np.array(testing_samples)
    X_train=training_samples[:,0]
    Y_train=training_samples[:,1]

    model_PolynomialFeature=PolynomialFeature()
    X_train=PolynomialFeature(degree=3).transform(X_train)
    model=LinearRegression()
    model.fit(X_train,Y_train)
    y_predict=model.predict(PolynomialFeature(degree=3).transform(testing_samples))
    for y in y_predict:
        print(y)