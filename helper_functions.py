#!/usr/bin/env python
# coding: utf-8

# In[49]:


import h5py
import numpy as np
import matplotlib.pyplot as plt



# ## Activation Functions (Forward & Backward)
# 

# In[51]:


def sigmoid (Z):

    
    A = 1/(1+np.exp(-Z))
    return A
def relu(x):
    A=np.maximum(0,x)
    return A
def softmax(Z):
    T=np.exp(Z)
    A=T/np.sum(T,axis=0,keepdims=True)
    return A
def activation(Z,fx):

    if (fx=="sigmoid"):
        return sigmoid(Z)
    elif (fx=="relu"):
         return relu(Z)
    elif((fx=="tanh")):
          return np.tanh(Z)    
    elif((fx=="softmax")):
        return softmax(Z) 
def tanh_backward(A):
    return (1-np.power(A, 2))
    
def relu_backward(Z):
    dZ=np.zeros(Z.shape)
    dZ[Z>0]=1
    return dZ
def sigmoid_backward(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ
def softmax_backward(Z):
    T=np.exp(Z)
    A=T/np.sum(T,axis=0)
    dZ=A*(1-A)
    return dZ


## output layer should be sigmoid
def loss_backward(Y,A,activation):
    if activation=="sigmoid":
        print("sigmoid detected")
        return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
    elif activation=="softmax":
        back_loss=-np.divide(Y,A)
        #print("Y = ",Y)
        #print("A = ",A)
       # print("back loss dim = ",back_loss)
        return back_loss
def backward(Z,fx):
    if(fx=="sigmoid"):
        return sigmoid_backward(Z)
    elif(fx=="relu"):
        return relu_backward(Z)
    elif(fx=="tanh"):
        return tanh_backward(Z)
    elif(fx=="softmax"):
        return softmax_backward(Z)


# ## Forward Propagation Functions
# perform one forward iteration through the network layers, this function takes W,b parameters in addition to the layers dimensions as inputs and outputs the linear, activation functions of each layer in the cache 
# 

# In[52]:


def forward_prop(X,params,l_param,keepprob):
    A_prev=X
    cache={"A0":X}
    L=len(l_param["dim"])
    for i in range(1,L+1):
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        Z=np.dot(W,A_prev)+b # z shape should be m*1 
        A_prev=activation(Z,l_param["activation"][i-1])
        if(i<L):
            Drop_mask=np.random.rand(A_prev.shape[0],A_prev.shape[1])
            keepprob=(1-keepprob)*(i/L) + keepprob
            Drop_mask=(Drop_mask < keepprob).astype(int)
            A_prev=np.multiply(A_prev,Drop_mask)
            A_prev= A_prev/keepprob
            cache["D"+str(i)]=Drop_mask
        cache["Z"+str(i)]=Z
        cache["A"+str(i)]=A_prev
    return cache


# ## Parameters Initializations

# In[53]:


def initialize_dnn(input_size,n,scale=.01):
    l_dims=[input_size]
    l_dims.extend(n["dim"])
    params={}
    opt_params={}
    for i in range(1,len(l_dims)) :
        params["W"+str(i)]=np.random.randn(l_dims[i],l_dims[i-1])* np.sqrt(2./l_dims[i-1])
        params["b"+str(i)]=np.zeros((l_dims[i],1))
        opt_params["VdW"+str(i)]=np.zeros((l_dims[i],l_dims[i-1]))
        opt_params["Vdb"+str(i)]=np.zeros((l_dims[i],1))
        opt_params["SdW"+str(i)]=np.zeros((l_dims[i],l_dims[i-1]))
        opt_params["Sdb"+str(i)]=np.zeros((l_dims[i],1))
    return params,opt_params

def compute_cost(A,Y,lambd,l_param,params):
    m=A.shape[1]
    L2=0
    activation=l_param["activation"][len(l_param["activation"])-1]
    if activation =="softmax":
        loss=np.sum(np.multiply(Y,-np.log(A)),axis=0,keepdims=True)
        #print("loss dim",loss)
        cost=(1./m)*np.sum(loss,axis=1)
    else:
        cost=(1./m)* np.nansum((np.multiply(Y ,-np.log(A)) + np.multiply((1-Y),-np.log(1-A))))
    for layer in range(1,len(l_param)+1):
        L2+=np.sum(np.square(params["W"+str(layer)]))
    L2=(lambd/(2*m))*L2
    cost = np.squeeze(cost) +L2
    return  cost#loss for all images b

# ### Backward Propagation Functions

# In[54]:


def layer_backward(cache,grads,W,activation,layer_ind,keepprob,lambd,Y):
    #print("layer index = ",layer_ind)
    if(activation!="softmax"):
        dA=grads["dA"+str(layer_ind)]
    A_prev=cache["A"+str(layer_ind-1)]
    A=cache["A"+str(layer_ind)]
    Z=cache["Z"+str(layer_ind)]
    m = A_prev.shape[1]
    if (activation=="tanh"):
        dZ= dA* backward(A,activation)
    elif(activation=="softmax"):
        dZ=A-Y
    else:
        dZ= dA* backward(Z,activation)
    dW =(1./m )* np.dot(dZ,A_prev.T)+ (lambd/m)*W
    db = (1./m )* np.sum(dZ, axis = 1, keepdims = True)
    dA_prev=np.dot(W.T,dZ)
    return dW,db,dA_prev   

def backward_prop(X,Y,cache,params,l_param,keepprob,lambd):
    L=len(l_param["dim"])
    grads={}
    if(l_param["activation"][L-1]!="softmax"):
        grads={"dA"+str(L):loss_backward(Y,cache["A"+str(L)],l_param["activation"][L-1])}
    for layer in range(L,0,-1):
        dW,db,dA_prev=layer_backward(cache,grads, params["W"+str(layer)],l_param["activation"][layer-1],layer,keepprob,lambd,Y)
        grads["dW" + str(layer)]=dW
        grads["db" + str(layer)]=db
        if (layer!=1):
            Drop_mask=cache["D"+str(layer-1)]
            dA_prev=np.multiply(dA_prev,Drop_mask)/keepprob
        grads["dA" + str(layer-1)]=dA_prev
    return grads


# ## Gradient Decent Iteration
# iterate(X,Y,params,l_param) function performs one gradient decent over the network by performing the forward propagation to compute A[i],Z[i],loss and then perform the backward propagation to calculate the derevatives of the layers parameters dW[i],db[i] and dA[i] by using the chain rule.
# After calcualting the gradients , these gradients are used later to update the parameters toward the direction that will decrease the loss.
# This function outputs cache that has the layers linear and activation functions, gradients of the layers parameters, and finally the cost that represents the total loss over all the training set.

# In[105]:




def iterate(X,Y,params,l_param,keepprob,lambd):
    L=len(l_param["dim"])
    cache=forward_prop(X,params,l_param,keepprob)
    
    cost=compute_cost(cache["A"+str(L)],Y,lambd,l_param,params)
    grads=backward_prop(X,Y,cache,params,l_param,keepprob,lambd)
    #print(cache["A3"][:,4])
    return cache,cost,grads     
    
##iterate for # of loops to decrease the cost which increases he accuracy
def optimize(X,Y,X_t,Y_t,params,opt_params,l_param,learning_rate,iterations,keepprob,lambd,batch_size,optimization,beta1,beta2,episilom):
    costs = []
    for i in range(iterations):
        batches=shuffle_and_partition(X,Y,batch_size)
        cost_epoch=0
        for batch in batches:
            batch_x=batch[0]
            batch_y=batch[1]
            cache,cost,grads=iterate(batch_x,batch_y,params,l_param,keepprob,lambd)
            params,opt_params=update_parameters(params,grads,opt_params,len(l_param),learning_rate,optimization,beta1,beta2,episilom)
            cost_epoch+=cost
        cost_epoch/=len(batches)
        #print(len(batches),"is detected")
        costs.append(cost_epoch)
        if i %50 ==0:
            print ("cost =",cost_epoch)
        if(i%200==0):
            train_acc=predict(X,Y,params,l_param)
            test_acc=predict(X_t,Y_t,params,l_param)
            print("train accuracy: {} %".format(train_acc))
            print("test accuracy: {} %".format(test_acc))
    return params,cache,costs

# def predict(X,Y,params,l_param):
#     A=X
#     m=X.shape[1] #number of images in the trainning set
#     Y_pred = np.zeros((1,m))    
#     for i in range(1,len(l_param["dim"])+1):
#         W=params["W"+str(i)]
#         b=params["b"+str(i)]
#         Z=np.dot(W,A)+b # z shape should be m*1
#         A=activation(Z,l_param["activation"][i-1])
#     for j in range(m):
#         if(A[0][j]>0.5):
#             Y_pred[0][j]=1
#         else:
#             Y_pred[0][j]=0
#     acc=(100 - np.mean(np.abs(Y_pred - Y)) * 100)   
#     return acc
def predict(X,Y,params,l_param):
    A=X
    m=X.shape[1] #number of images in the trainning set
    L=len(l_param["dim"])
    #Y_pred = np.zeros(Y.shape)    
    for i in range(1,L+1):
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        Z=np.dot(W,A)+b # z shape should be m*1
        A=activation(Z,l_param["activation"][i-1])
    if l_param["activation"][L-1]=="softmax":
        A_max=np.amax(A,axis=0,keepdims=True)
        Y_pred=(A>=A_max).astype(np.int)
        Y_pred=Y_pred.astype(np.float)
        absar=np.sum(np.abs(Y_pred-Y),axis=0)/2.0
        mean=np.mean(absar)
        #print("mean = ",mean)
        acc=(100 - mean*100)   
    else :
        for j in range(m):
            if(A[0][j]>0.5):
                Y_pred[0][j]=1
            else:
                Y_pred[0][j]=0
        acc=(100 - np.mean(np.abs(Y_pred - Y)) * 100)   
    return acc

# In[ ]:





# In[98]:


#model for iterattion, optimization ,calculating accuracy
def model (X_train,Y_train,X_test,Y_test,layer_param,learning_rate=0.01,iterations=4000,keepprob=1,lambd=0.0,batch_size=64,optimization="None",beta1=0.9,beta2=0.999,episilom=10e-8):
    params,opt_params=initialize_dnn(X_train.shape[0],layer_param)
    params,cache,costs=optimize(X_train,Y_train,X_test,Y_test,params,opt_params,layer_param,learning_rate,iterations,keepprob,lambd,batch_size,optimization,beta1,beta2,episilom)
    acc_train=predict (X_train,Y_train,params,layer_param)
    acc_test=predict (X_test,Y_test,params,layer_param)
    print("train accuracy: {} %".format(acc_train))
    print("test accuracy: {} %".format(acc_test))
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return params,cache


def training_set_shuffle(X,Y):
    training_data=np.row_stack((X,Y))
    np.take(training_data,np.random.permutation(training_data.shape[1]),axis=1,out=training_data)
    return training_data

def training_set_partition(batch_size,training_data,y_shape):
    X=training_data[:-y_shape,:]
    Y=training_data[-y_shape:,:]
    batches=[]
    m=X.shape[1]

    number_of_batches=int(m/batch_size)
    for batch in range(1,number_of_batches+1):
        batch_x=X[:,(batch-1)*batch_size:batch*batch_size]
        batch_y=Y[:,(batch-1)*batch_size:batch*batch_size]
        batches.append((batch_x,batch_y))
    if(m % batch_size !=0):
        last_batch_x=X[:,number_of_batches*batch_size:]
        last_batch_y=Y[:,number_of_batches*batch_size:]
        batches.append((last_batch_x,last_batch_y))
    return batches
def shuffle_and_partition(X,Y,batch_size):
    training_data=training_set_shuffle(X,Y)
    batches=training_set_partition(batch_size,training_data,Y.shape[0])
    return batches
def update_parameters(params,grads,opt_params,layers_number,learning_rate,optimization,beta1,beta2,episilom):

    if optimization=="Momentum":
        for layer in range(1,layers_number+1):
            opt_params["VdW"+str(layer)] = beta1 * opt_params["VdW"+str(layer)] + (1-beta1) * grads["dW"+str(layer)]
            opt_params["Vdb"+str(layer)] = beta1 * opt_params["Vdb"+str(layer)] + (1-beta1) * grads["db"+str(layer)]
            params["W"+str(layer)] -= learning_rate* opt_params["VdW"+str(layer)]
            params["b"+str(layer)] -= learning_rate* opt_params["Vdb"+str(layer)]
    elif optimization == "RMSprop" :
        for layer in range(1,layers_number+1):
            opt_params["SdW"+str(layer)] = beta2 * opt_params["SdW"+str(layer)] + (1-beta2) * np.square(grads["dW"+str(layer)])
            opt_params["Sdb"+str(layer)] = beta2 * opt_params["Sdb"+str(layer)] + (1-beta2) * np.square(grads["db"+str(layer)])
            params["W"+str(layer)] -= learning_rate* (np.divide(grads["dW"+str(layer)],np.sqrt(opt_params["SdW"+str(layer)])))
            params["b"+str(layer)] -= learning_rate* (np.divide(grads["db"+str(layer)],np.sqrt(opt_params["Sdb"+str(layer)])))
    elif optimization=="Adam":
        for layer in range(1,layers_number+1):
            opt_params["VdW"+str(layer)] = beta1 * opt_params["VdW"+str(layer)] + (1-beta1) * grads["dW"+str(layer)]
            opt_params["Vdb"+str(layer)] = beta1 * opt_params["Vdb"+str(layer)] + (1-beta1) * grads["db"+str(layer)]
            opt_params["SdW"+str(layer)] = beta2 * opt_params["SdW"+str(layer)] + (1-beta2) * np.square(grads["dW"+str(layer)]+episilom)
            opt_params["Sdb"+str(layer)] = beta2 * opt_params["Sdb"+str(layer)] + (1-beta2) * np.square(grads["db"+str(layer)]+episilom)
            params["W"+str(layer)] -= learning_rate * np.divide(opt_params["VdW"+str(layer)],np.sqrt(opt_params["SdW"+str(layer)]+episilom))
            params["b"+str(layer)] -= learning_rate * np.divide(opt_params["Vdb"+str(layer)],np.sqrt(opt_params["Sdb"+str(layer)]+episilom))
    elif optimization=="None":
        for layer in range(1,layers_number+1):
            params["W"+str(layer)] -= learning_rate* grads["dW"+str(layer)]
            params["b"+str(layer)] -= learning_rate* grads["db"+str(layer)]
    return params,opt_params
def test(X,params,l_param):
    A=X
    L=len(l_param["dim"])
    for i in range(1,L+1):
        W=params["W"+str(i)]
        b=params["b"+str(i)]
        Z=np.dot(W,A)+b # z shape should be m*1
        A=activation(Z,l_param["activation"][i-1])
    if l_param["activation"][L-1]=="softmax":
        A_max=np.amax(A,axis=0,keepdims=True)
        if(np.sum(A_max)==0):
            print("zero error")
        Y_pred=(A>=A_max).astype(np.int)
    else :
        for j in range(m):
            if(A[0][j]>0.5):
                Y_pred[0][j]=1
            else:
                Y_pred[0][j]=0 
    return Y_pred
    












