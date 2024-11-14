    # -*- coding: utf-8 -*-
    """
    Created on Wed Nov 13 14:48:24 2024
    
    @author: amr.shafek
    """
    
    import numpy as np 
    from sklearn.model_selection import train_test_split
    
    data = np.load("mnist_train_small.npy")
    print(data.shape)
    x = data[:,1:]
    # This means y contains all rows of just the first column, which is typically the label or target value for each example in data.
    y = data[:,0]
    
    
    X_train,X_test,yTrain,yTest = train_test_split(x,y,test_size=0.33,random_state=42)
    
    class CustomKNN:
        #first our instractor 
        def __init__(self,n_neighbours=5):
            self.n_neighbours = n_neighbours
            
        #Training Function 
        def fit(self,X,Y):
            self._X=X.astype(np.int64)
            self._Y=Y
        
        #predect point (by given a point and calculate which class it belongs to)
        def predect_point(self,point):
            list_Dic_ALL_Points_after_Calc=[]
            
            for x_points,y_points in zip(self._X,self._Y):
                Dic_points=((point - x_points)**2).sum()
                list_Dic_ALL_Points_after_Calc.append([Dic_points,y_points])
                print(list_Dic_ALL_Points_after_Calc)
            sorted_dic = sorted(list_Dic_ALL_Points_after_Calc)    
            top_K = sorted_dic[:self.n_neighbours]
            item ,count = np.unique(np.array(top_K)[:,1],return_counts=True)
            ans = item[np.argmax(count)]
            return ans
        
        #predect give me an aswer for each number in the array 
        def predect(self,X):
            result=[]
            for point in X :
                result.append(self.predect_point(point))
            return np.array(result,dtype=int)
        
        #return my score of the right answers my code found 
        def score(self,X,Y):
            sum(self.predect(X)==Y)/len(Y)
    
    
    
    
    
    print("HELLO")
    
    model =CustomKNN()
    model.fit(X_train, yTrain)
    print(model.predect(X_test[:10]))
    print(yTest[:10])
    
    print(model.score(X_test[:10], yTest[:10]))
