# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:59:44 2019

@author: rochej
"""
'CGAN with categorical conditions: VIX'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels
import statsmodels.api as sm

import keras.backend as K

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM, multiply, concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv1D,UpSampling1D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from functools import partial

from scipy import stats

import warnings
warnings.filterwarnings("ignore")

#parametre
z_size=100
label_size=1
n_critic = 5
batch_size=10
epoch=2000
#optimizer = RMSprop(lr=0.00005)
optimizer = Adam(lr=0.00005)


window=400
w_roll=10


#num_classes=43


#lecture data
ret_data_ini = pd.read_csv(filepath_or_buffer="ret_data.csv")

df=pd.read_excel('VIX.xlsx',index_col='Date')
dft=df.T
df_vol=df['VIX']
df_sp=df['SPX 500']



def get_return(Y):
    R = Y.pct_change()[1:]
    return(R)
    
def get_norm(data):
    mu=data.mean()
    sig=data.std()
    return((data-mu)/sig)
    

def split(X,w,roll_w):
    Xn=X.copy()
    Xn=np.array(Xn)
    
    L=[]
    for k in range(0,len(X),roll_w):
        if k+w<len(Xn):
            L.append(Xn[k:k+w])
        else:
            break
    L=pd.DataFrame(np.array(L))
    return(L)
    
def create_labelised(data):
    data_mat=np.array(data)
    label_vol=data_mat[:,0]
    label_vol=label_vol.reshape(-1, 1)
    return(label_vol)
    

    
def create_labelised_list(data,n_day):
    data_mat=np.array(data)
    label=data_mat[0:len(data)-1,0:n_day]
    return(label)
    
    
def get_price(returns,p_ini):
    prices=p_ini*(returns+1).cumprod()
    return(prices)

'data pr les entier'
data_vol=split(df_vol,window,w_roll)
label_r=create_labelised(data_vol)

df_t=data_vol.T
ret=get_return(df_t)
ret=ret.T
ret_data=ret

num_classes=len(label_r)



    


'pour les label en liste'

#data_vol_new=data_vol.iloc[1:]
#lab=create_labelised_list(data_vol,20) 
#num_classes=len(lab)
#
#df_t=data_vol_new.T
#ret=get_return(df_t)
#ret=ret.T
#ret_data=ret



#def build_generator():
#
#    
#    Z_input=Input(shape=(z_size,))
#    
#    output=Dense(128 * ret_data.shape[1] , activation="relu")(Z_input)
#    output=Reshape((ret_data.shape[1] ,128))(output)
#    
#    output=Conv1D(64, kernel_size=20, padding="same")(output)
#    output=BatchNormalization(momentum=0.8)(output)
#    output=Activation("elu")(output)
#    
#    output=Conv1D(32, kernel_size=20, padding="same")(output)
#    output=BatchNormalization(momentum=0.8)(output)
#    output=Activation("elu")(output)
#    
#    output=Conv1D(1, kernel_size=20, padding="same")(output)
#    output=Activation("linear")(output)
#    
#    label = Input(shape=(1,), dtype='float64')
#    label_embedding = Flatten()(Embedding(num_classes, z_size)(label))
#    
#    model_input = multiply([Z_input, label_embedding])
#    model=Model(model_input,output)
#    model.summary()
#    
#    return(model)
    
def build_generator():

    model = Sequential()

    model.add(Dense(200,activation="relu", input_dim=z_size))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(400,activation="elu"))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Dense(ret_data.shape[1],activation="linear"))
#    model.add(LeakyReLU(alpha=0.2))
#    model.add(Reshape(ret_data.shape[1]))

    model.summary()

    noise = Input(shape=(z_size,))
    label = Input(shape=(label_size,), dtype='float64')
    label_embedding = Flatten()(Embedding(num_classes, z_size)(label))
#    label_embedding=label
    
    model_input = multiply([noise, label_embedding])
    output = model(model_input)

    return Model([noise, label], output)

'test embedding'
#num_classes=60
#latent_dim=z_size
#print(latent_dim)
#   
#label = Input(shape=(20,), dtype='float64')
#label_embedding = (Embedding(num_classes, 100)(label))
#model=Model(label,label_embedding)  
#sampled_labels=np.array([1 for i in range(20)]).reshape(-1, 1) .T
#sampled_labels=lab[0,:].reshape(-1, 1).T


#x1=Input(shape=(20,), dtype='float64')
#x2=Input(shape=(30,), dtype='float64')
#output=concatenate([x1,x2])
#modelc=Model([x1,x2],output)
#
#X1=np.array([1 for i in range(20)]).reshape(1,20)
#X2=np.array([2 for i in range(30)]).reshape(1,30)
#result=modelc.predict([X1,X2])

  
#result=model.predict(sampled_labels)
    
#def build_generator():
#    
#    Z_input=Input(shape=(z_size,))
#    print(Z_input)
#    
#    output=Dense(128 , activation="relu",name="dense_1")(Z_input)
#    output=Dense(20, activation="elu",name="dense_2")(output)
#    output=Dense(5, activation="linear",name="dense_3")(output)
#  
#    output=Reshape((ret_data.shape[1] ,1))(output)
#    
#    
#
#    
#    model=Model(Z_input,output)
#    
#    model.summary()
#    return(model)
#    
    
#def build_discriminator():
#
#    
#    X_input=Input(shape=(ret_data.shape[1],1))
#    
#    output=Conv1D(16, kernel_size=20, strides=2, input_shape=(ret_data.shape[1],1), padding="same")(X_input)
#    output=LeakyReLU(alpha=0.2)(output)
#    
#    output=Conv1D(32, kernel_size=20, strides=2, padding="same")(output)
#    output=BatchNormalization(momentum=0.8)(output)
#    output=LeakyReLU(alpha=0.2)(output)
#    
#    output=Conv1D(64, kernel_size=20, strides=2, padding="same")(output)
#    output=BatchNormalization(momentum=0.8)(output)
#    output=LeakyReLU(alpha=0.2)(output)
#    
#    output=Conv1D(128, kernel_size=20, strides=1, padding="same")(output)
#    output=BatchNormalization(momentum=0.8)(output)
#    output=LeakyReLU(alpha=0.2)(output)
#    
#    output=Flatten()(output)
#    output=Dense(1)(output)
#    
#    label = Input(shape=(1,), dtype='float64')
#    label_embedding = Flatten()(Embedding(num_classes, ret_data.shape[1])(label))
#    
#    model_input = multiply([X_input, label_embedding])
#    model=Model(model_input, output)
#            
#    model.summary()
#    
#    return(model)  
    
def build_discriminator():

    model = Sequential()

    model.add(Dense(512, input_dim=ret_data.shape[1]))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    X_input=Input(shape=(ret_data.shape[1],))

    label = Input(shape=(label_size,), dtype='float64')
    label_embedding = Flatten()(Embedding(num_classes, ret_data.shape[1])(label))

    model_input = multiply([X_input, label_embedding])

    validity = model(model_input)

    return Model([X_input, label], validity)
    

            
def train(X,label_r):
    num_train = X.shape[0]
    start = 0
    E=[]

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for e in range(epoch):
        for _ in range(n_critic):
            stop = start + batch_size
            real_batch = X[start:stop]
            real_batch=np.array(real_batch)
#            real_batch=real_batch.reshape((real_batch.shape[0],real_batch.shape[1]))
            noise =np.random.normal(0, 0.1, size=(batch_size, z_size))
            label_batch_r=label_r[start:stop]
            
            d_loss_real = discriminator.train_on_batch([real_batch, label_batch_r], valid)
            gen_imgs = generator.predict([noise, label_batch_r])
            d_loss_fake = discriminator.train_on_batch([gen_imgs, label_batch_r], fake)
##            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
##            
        g_loss = combined.train_on_batch([noise, label_batch_r], valid)
        start += batch_size
        if start > num_train - batch_size:
            start = 0
        if e % 10 == 0:
#            print(d_loss[-1],d_loss[0],d_loss[1])1
            E.append(d_loss)
            print(d_loss)
            print(e)
            
#        if e % 400 == 0 and e>0:
#            label_t=label_r[0].reshape(-1,1)
#            n_gene=1
#            noise_t =np.random.normal(0, 0.1, size=(n_gene,z_size))
##noise_t=np.array([1 for i in range(100)]).reshape(1,100)
##noise_t=ret_real.iloc[:100]
##noise_t=np.array(noise_t).reshape(1,len(noise_t))
#
#            ret_gene=generator.predict([noise_t,label_t])
#            ret_gene=pd.DataFrame(ret_gene).T
#            price_gene=get_price(ret_gene,label_t)
#            plt.plot(price_gene,label=str(e))
            
            

'modele simple'
discriminator =build_discriminator()
#discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer)
generator = build_generator()
#
label = Input(shape=(label_size,))
noise = Input(shape=(z_size, ))
#
noise_gen = generator([noise,label])
#
discriminator.trainable = False
#
valid = discriminator([noise_gen, label])
combined = Model([noise, label], valid)
combined.compile(loss=['binary_crossentropy'],optimizer=optimizer)

print('ok')


epoch=5000
train(ret_data,label_r)


def simulate(i,N_sim):
    label_t=label_r[i].reshape(-1,1)

    L=[]
    for k in range(N_sim):
        n_gene=1
        noise_t =np.random.normal(0, 0.1, size=(n_gene,z_size))
        ret_gene=generator.predict([noise_t,label_t])
        ret_gene=pd.DataFrame(ret_gene).T
        price_gene=get_price(ret_gene,label_t)
        plt.plot(price_gene,label='gene')
        L.append(price_gene)
    X=pd.concat([L[i] for i in range(len(L))],axis=1)
    X_mean=X.mean(axis=1)
    
    price_real=data_vol.iloc[i]
    plt.plot(X_mean,label='moy')    
    plt.plot(price_real,label='reel')
    plt.ylabel('VIX level')
    plt.xlabel('time')
    plt.legend()
    return(X_mean,price_real)

p_g,p_r=simulate(209,5)
p_r=p_r.iloc[1:]

r_g=get_return(p_g)
r=get_return(p_r)

'QQ plot'

plt.figure()
plt.scatter(np.sort(r), np.sort(r_g))
x=np.array([-1,1])
y=np.array([-1,1])
plt.plot(x,y)
plt.xlabel('reel')
plt.ylabel('gene')

'ACF et PCF'
def aff_autocorr(n_lag,s,leg):
    A=[]

    for k in range(1,n_lag+1):
        A.append(s.autocorr(lag=k))

    Ad=pd.Series(A)
    x=np.array([i for i in range(1,n_lag+1)])
    plt.plot(x,A,label=str(leg))
    plt.xlabel('lag')
    plt.ylabel('ACF')
#    plt.plot(Ad)
    
    return(Ad)

plt.figure()    
aff_autocorr(60,r,'real')
aff_autocorr(60,r_g,'fake')
plt.legend()




def get_correlation(x,y,start,end):
    s1=x.iloc[start:end]
    s2=y.iloc[start:end]

    return(s1.corr(s2,method='pearson'))
    
def get_mean_corr(x,y,roll_w):
    R=[]
    for start in range(0,len(x),5):
        if start+roll_w > len(x):
            rho=get_correlation(x,y,start,len(x))
            R.append(rho)
        else:
            rho=get_correlation(x,y,start,start+roll_w)
            R.append(rho)
    corr_m=sum([rho for rho in R])
    print(corr_m)
    
#price_real=pd.Series(price_real)
#price_gene.columns=['gene']
#price_gene=price_gene['gene']
#
#def get_mean_corr(roll_w):
#    
#    x=price_gene
#    y=price_real
#    R=[]
#    for start in range(0,len(x),5):
#        if start+roll_w > len(x):
#            rho=get_correlation(x,y,start,len(x))
#            R.append(rho)
#        else:
#            rho=get_correlation(x,y,start,start+roll_w)
#            R.append(rho)            
#    
#    rho_mean=np.mean(R)
#    print(rho_mean)
#
#print(get_mean_corr(price_real,price_gene,20))
#
#print(get_correlation(price_real,price_gene,60,80))



#s1=pd.Series(price_real).iloc[60:80]
#price_gene.columns=['gene']
#s2=price_gene['gene'].iloc[60:80]
#print(s1.corr(s2))
#s2=price_gene['gene']



#ret_gene=ret_gene.reshape((526,ret_data.shape[1]))
#ret_gene=pd.DataFrame(ret_gene).T
#plt.plot(ret_gene,marker='o')
#ret_mean=ret_data.mean(axis=0)
#plt.plot(ret_mean)





def generate_data(stock_name,aff_return=False,aff_histo=False):
    
    noise =np.random.normal(0, 0.1, size=(n_gene, z_size))
    ret_gene=generator.predict(noise)
    
    ret_gene=ret_gene.reshape((n_gene,ret_data.shape[1]))
    ret_gene=pd.DataFrame(ret_gene)
    ret_gene.columns=ret_data.columns
    
    
    ret_gene_norm=get_norm(ret_gene)
    ret_data_norm=get_norm(ret_data)
    
    if aff_return:
        plt.figure(1)
    
        R=pd.concat([ret_gene_norm[stock_name],ret_data_norm[stock_name]],axis=1)
        R.columns=['gene','vrai']
        plt.plot(R)
        
    if aff_histo:
        plt.figure(2)
        ret_gene_norm[stock_name].hist(normed=True,bins=80,alpha = 0.5)
        ret_data_norm[stock_name].hist(normed=True,bins=80,alpha = 0.5)

    
    
    return(ret_gene[stock_name])
    
#generate_data('Apple',aff_return=True,aff_histo=True) 
    
    
def mean_generate(stock_name,N_mean,aff_return=False,aff_histo=False):
    L=[]
    for _ in range(N_mean):
        x_gene=generate_data(stock_name)
        L.append(x_gene)
    X=pd.DataFrame(L).T
    X_mean=X.mean(axis=1)
    
    X_mean_norm=get_norm(X_mean)
    ret_data_norm=get_norm(ret_data)
    
    
    if aff_return:
        plt.figure(1)
    
        R=pd.concat([X_mean_norm,ret_data_norm[stock_name]],axis=1)
        R.columns=['gene_mean','vrai']
        plt.plot(R)
        plt.legend()
#        R.plot()
        
        
    if aff_histo:
        plt.figure(2)
        X_mean_norm.hist(normed=True,bins=80,alpha = 0.5)
        ret_data_norm[stock_name].hist(normed=True,bins=80,alpha = 0.5)
        
    
        
    return(X_mean)
    
#X=mean_generate('Apple',10)
    
   
    
def KS_test(X,Y,alpha=0.1):
    x=np.array(X)
    y=np.array(Y)
    test=stats.ks_2samp(x, y)
    p_value=test[1]
    if p_value<alpha:
        print('not same distribution')
    else:
        print('same distribution')
    return(p_value)

#ret_data=ret_data_ini
#x=generate_data('Apple') 
#
#
#test1=KS_test(x,ret_data['Apple'])
#test2=KS_test(ret_data['Intel'],ret_data['Box'])

#print(test1)   
    
#sampled_labels = np.arange(0, 10).reshape(-1, 1)    