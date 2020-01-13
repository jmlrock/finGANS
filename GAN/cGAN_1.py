
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:58:15 2019

@author: rochej
"""
'CGAN avec une condition non categrical: liste: paper 1 avec scaling zscore'

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
from keras.optimizers import RMSprop
from functools import partial

from scipy import stats


import warnings
warnings.filterwarnings("ignore")

#parametre
z_size=100
label_size=1
n_critic = 5

epoch=20
optimizer = RMSprop(lr=0.00005)




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
    

def split(X,w,roll_w,label_w):
    
    Xn=X.copy()
    Xn=np.array(Xn)
    
    label=[]
    L=[]
    for k in range(label_w,len(X),roll_w):
        if k+w<len(Xn):
            L.append(Xn[k:k+w])
            label.append(Xn[k-label_w:k])
        else:
            break
            
            
    label=pd.DataFrame(np.array(label))
    L=pd.DataFrame(np.array(L))
    return(L,label)
    
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
    
def cumreturn(ret):
    return((1+ret).cumprod())
    
def zscore(series):
    return (series - series.mean()) / np.std(series)


window=1000
w_roll=100
label_w=250
batch_size=250

#data_vol=split(df_vol,window,w_roll,label_w)[0]
#label=split(df_vol,window,w_roll,label_w)[1]


'normalize data'
#df_t=data_vol.T 
#p_ini=df_t.iloc[0]
#ret=get_return(df_t)
#ret=ret.T
#ret_data=ret

#l_t=label.T
#lab=get_return(l_t)
#lab=lab.T
#lab_data=lab

'input: zscore des returns du SP500'
label_size=label_w
zscore_sp=zscore(get_return(df_sp)) #zscore des return du SPX
m_data=get_return(df_sp).mean()
sig_data=get_return(df_sp).std()
ret_data=np.array(zscore_sp).reshape(-1,1) #ret_data: nom de l'input dans tout le reseau
ret=cumreturn(get_return(df_sp)) #cumulative return du SPX

#plt.plot(ret)
#plt.plot(zscore(ret))

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
'le bon gene'    
def build_generator():
    

    model = Sequential()

    model.add(Dense(200, input_dim=z_size+label_size))
#    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(100))
#    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(50))
#    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(25))
#    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(ret_data.shape[1],activation='linear')) #return un scalaire; y_t
#    model.add(LeakyReLU(alpha=0.5))
#    model.add(Reshape(ret_data.shape[1]))

    model.summary()

    noise = Input(shape=(z_size,))
    label = Input(shape=(label_size,))
    
    model_input = concatenate([noise, label])
    output = model(model_input)

    return Model([noise, label], output)
'test gene conv'
#def build_generator():
#    model = Sequential()
#
#    model.add(Dense(128 * ret_data.shape[1], input_dim=z_size+label_size))
#    model.add(LeakyReLU(alpha=0.5))
#    model.add(Reshape((ret_data.shape[1] ,128)))
#    model.add(Conv1D(64, kernel_size=4, padding="same"))
#    model.add(LeakyReLU(alpha=0.5))
#    model.add(Conv1D(32, kernel_size=4, padding="same"))
#    model.add(LeakyReLU(alpha=0.5))
#    model.add(Conv1D(1, kernel_size=4, padding="same"))
#    model.add(Activation("linear"))
#   
#    model.summary()
#    noise = Input(shape=(z_size,))
#    label = Input(shape=(label_size,))
#    
#    model_input = concatenate([noise, label])
#    output = model(model_input)
#
#    return Model([noise, label], output)

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
    
    
#def build_discriminator():
#
#    model = Sequential()
#
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
'le bon disc'    
def build_discriminator():

    model = Sequential()

    model.add(Dense(100, input_dim=ret_data.shape[1]+label_size))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.5))
#    model.add(Dropout(0.4))
    
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.5))
    
#    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid')) #cgan classique:output is a proba: sigmoid ou softmax 
    model.summary()
    
    X_input=Input(shape=(ret_data.shape[1],))
    label = Input(shape=(label_size,))

    model_input = concatenate([X_input, label])

    validity = model(model_input)

    return Model([X_input, label], validity)

'test disc conv'
#def build_discriminator():
#    model = Sequential()
#    
#    model.add(Dense(128, input_dim=ret_data.shape[1]+label_size))
#    model.add(LeakyReLU(alpha=0.5))
#    model.add(Reshape((ret_data.shape[1] ,128)))
#
#    model.add(Conv1D(64, kernel_size=4, padding="same"))
#    model.add(LeakyReLU(alpha=0.5))
#    model.add(Conv1D(32, kernel_size=4, padding="same"))
#    model.add(LeakyReLU(alpha=0.5))
#    model.add(Conv1D(1, kernel_size=4, padding="same"))
#    model.add(Activation("sigmoid"))
#    
#    model.summary()
#    
#    X_input=Input(shape=(ret_data.shape[1],))
#    label = Input(shape=(label_size,))
#
#    model_input = concatenate([X_input, label])
#
#    validity = model(model_input)
#
#    return Model([X_input, label], validity)

    
    

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


#def train_W(X,label_r):
#    num_train = X.shape[0]
#    start = 0
#    valid = -np.ones((batch_size, 1))
#    fake =  np.ones((batch_size, 1))
#    
#    for e in range(epoch):
#        for _ in range(n_critic):
#            stop = start + batch_size
#            real_batch = X[start:stop]
#            real_batch=np.array(real_batch)
#            real_batch=real_batch.reshape((real_batch.shape[0],real_batch.shape[1],1))
#            noise =np.random.normal(0, 0.1, size=(batch_size, z_size))
#            label_batch_r=label_r[start:stop].reshape(-1, 1)
#            d_loss = discriminator_model.train_on_batch([real_batch, noise, label_batch_r],[valid, fake])
#            
#        g_loss = generator_model.train_on_batch([noise,label_batch_r], valid)
#        start += batch_size
#        if start > num_train - batch_size:
#            start = 0
#        if e % 10 == 0:
##            print(d_loss[-1],d_loss[0],d_loss[1])
#            print(e)
##            print('epoch: {}; D loss: {:.4}; G_loss: {:.4}'.format(e, d_loss[-1], g_loss))
        
    
            
def train(X):
    num_train = X.shape[0]
    start = label_w
    E=[]

#    label associé nécessaire pour le calcul de la cross entropy
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for e in range(epoch):
        for _ in range(n_critic):
            stop = start + batch_size
            real_batch = X[start:stop]
            label_batch=[]
            for k in range(start,stop):
                label=X[k-label_w:k] #fenetre des n jours precedents
                label_batch.append(label)
            label_batch=np.array(label_batch)
                
            real_batch=np.array(real_batch)
            label_batch=np.array(label_batch)
            label_batch=label_batch.reshape(batch_size,label_w)

    
            real_batch=real_batch.reshape((real_batch.shape[0],real_batch.shape[1]))
            noise =np.random.normal(0, 0.5, size=(batch_size, z_size))

            
            d_loss_real = discriminator.train_on_batch([real_batch, label_batch], valid) #input: real data
            gen_imgs = generator.predict([noise, label_batch])
            d_loss_fake = discriminator.train_on_batch([gen_imgs, label_batch], fake) #input: fake data
##            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
##            
        g_loss = combined.train_on_batch([noise, label_batch], valid)
        start += batch_size
        if start > num_train - batch_size:
            start = label_w
        if e % 10 == 0:
#            print(d_loss[-1],d_loss[0],d_loss[1])
            E.append(d_loss)
            print('d_loss_tot=',d_loss)
            print('fake=',d_loss_fake)
            print('real=',d_loss_real)
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
##
label = Input(shape=(label_size,))
noise = Input(shape=(z_size, ))
##
noise_gen = generator([noise,label])
#noise_gene_r=Reshape((1,1))(noise_gen)
#label_r=Reshape((1,250))(label)


##
discriminator.trainable = False
##
valid = discriminator([noise_gen, label])
combined = Model([noise, label], valid)
combined.compile(loss=['binary_crossentropy'],optimizer=optimizer)

print('ok')
#train(ret_data,label_r)

'model de Wassertein'
#generator=build_generator()
#discriminator=build_discriminator()
#
#
#
##def du label
#label = Input(shape=(1,))
#
##graph de generateur
#
#discriminator.trainable = False
#generator.trainable = True
#        
#noise_gen = Input(shape=(z_size, ))
#
#
#noise_data = generator([noise_gen,label])
#valid = discriminator([noise_data,label])
#
#generator_model = Model([noise_gen,label], valid)
#generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)
#
#
#
#
##graph du discirminateur
#
#generator.trainable = False
#
#real_data = Input(shape=(ret_data.shape[1],1))
#noise = Input(shape=(z_size,))
#fake_data = generator([noise,label])
#
#fake = discriminator([fake_data,label])
#valid = discriminator([real_data,label])
##
#discriminator_model = Model(inputs=[real_data, noise,label], outputs=[valid, fake])
#discriminator_model.compile(loss=[wasserstein_loss, wasserstein_loss], optimizer=optimizer, loss_weights=[1, 1])
#
#
#    

epoch=1000
train(ret_data)
#    
#
##ret_real=get_return(df_vol)
#ret=cumreturn(get_return(df_sp))
##
def generate_serie():
    R=[]
#    noise_t =np.random.normal(0, 1, size=(n_gene,z_size))
    for k in range(label_w,len(ret_data)):
        
        label_t=ret_data[k-label_w:k]
        label_t=label_t.reshape(1,len(label_t))

        n_gene=1
        noise_t =np.random.normal(0.5, 0.5, size=(n_gene,z_size))
    #    noise_t =np.random.normal(0, 0.01, size=(n_gene,z_size))
    ##noise_t=np.array([1 for i in range(100)]).reshape(1,100)
    ##noise_t=ret_real.iloc[:100]
    ##noise_t=np.array(noise_t).reshape(1,len(noise_t))
    #
        ret_gene=generator.predict([noise_t,label_t])
    #    ret_gene=pd.DataFrame(ret_gene).T
        R.append(float(ret_gene))
        
    R=pd.DataFrame(R) #R: output du generateur
    R_real=pd.DataFrame(ret_data[label_w:]) #rett_data: zscore des return du SPX
    
    ret_gene_brut=R #output brut 
    ret_gene=R*sig_data+m_data #rescaling pour obtenir des data reels: dezcscorise avec les données des data reel
    ret_real=R_real*sig_data+m_data #de-zscorise ret_data: get the real cumreturn
    
    plt.figure(1)
    plt.plot(cumreturn(ret_gene),label='cumreturn gene')
    plt.plot(cumreturn(ret_real),label='cumreturn reel')     
    plt.legend()     

    plt.figure(2)
    plt.plot(zscore(cumreturn(ret_gene)),label='z_s(cumreturn gene)')
    plt.plot(zscore(cumreturn(ret_real)),label='z_s(cumreturn reel)')     
    plt.legend()  
    
    plt.figure(3)
    plt.plot(ret_gene,label='return gene')
    plt.plot(ret_real,label='return real')
    plt.legend()
    
    plt.figure(4)
    plt.plot(ret_gene_brut,label='return gene brut')
    plt.plot(R_real,label='ret_data')
    plt.legend()
    
    plt.figure(5)
    plt.plot(cumreturn(R),label='output brut')

    
    
generate_serie()


def simulate_data(n_sim):
    for k in range(n_sim):
        data=generate_serie()
        data_gene=data[1]
        data_reel=data[0]
        print(k)
        plt.plot(data_gene,label='gene_nn')
    plt.plot(data_reel,label='reel')
    plt.legend()
    
#simulate_data(1) 

def simulate_mean_data(n_sim):
    L=[]
    for k in range(n_sim):
        data=generate_serie()
        data_gene=data[1]
        data_reel=data[0]
        L.append(data_gene)
    X=pd.concat([L[i] for i in range(len(L))],axis=1)
    X_mean=X.mean(axis=1)
    plt.plot(X_mean,label='gene')  
    plt.plot(data_reel,label='reel')
    plt.legend()
    
#simulate_mean_data(10)


#plt.plot(price_gene,label='gene')
#plt.plot(price_real,label='real')
#plt.legend()

#plt.figure(1)
#plt.plot(R,label='gene')
#plt.legend()
#
#plt.figure(2)
#plt.plot(R_real,label='real')
#plt.legend()

###
#price_real=data_vol.iloc[0]
##
#plt.figure(3)
#plt.plot(price_gene,label='gene')
#plt.plot(price_real,label='reel')
#plt.legend()

#R_new=R-R.mean()
#prix_new=get_price(R_new,1).iloc[:500]
#plt.plot(prix_new,label='ge')
#plt.plot(price_real,label='real')

#plt.figure(4)
#plt.plot(R,label='gene')
#plt.plot(R_real,label='real')
#plt.legend()



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

