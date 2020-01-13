# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 17:52:52 2019

@author: rochej
"""

'Wassertein cGAN with continuous labelling: p past days'


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

from sklearn.preprocessing import MinMaxScaler

from statsmodels.graphics.tsaplots import plot_acf


import warnings
warnings.filterwarnings("ignore")

#parametre
z_size=500
n_critic = 5
label_w=100

optimizer = RMSprop(lr=0.00005)

'non constant learning rate'
#learning_rate=0.001
#decay_rate=learning_rate/epoch


#optimizer = RMSprop(lr=learning_rate,decay=decay_rate)
#optimizer = RMSprop(lr=learning_rate)
#optimizer = Adam(lr=learning_rate,decay=decay_rate)


#window=1000
#w_roll=1000
#label_w=300
batch_size=30


#lecture data
ret_data_ini = pd.read_csv(filepath_or_buffer="ret_data.csv")

df=pd.read_excel('VIX.xlsx',index_col='Date')
dft=df.T
df_vol=df['VIX']

D_sp=df['SPX 500']
df_sp=np.array(df['SPX 500'])
df_vol=np.array([df['VIX']])
#plt.plot(df_sp)


scaler = MinMaxScaler(copy=False)
ret_data = scaler.fit_transform(df_sp.reshape(-1, 1))
#ret_data = scaler.fit_transform(df_vol.reshape(-1, 1))
#ret_ini=scaler.inverse_transform(ret_data)


#ret_data = scaler.fit_transform(df_vol.reshape(-1, 1))
#ret_origine=scaler.inverse_transform(ret_data)



label_size=label_w

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated data samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((batch_size,1)) #(batch_size,col,1)
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])
    
    

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    
        #Computes gradient penalty based on prediction and weighted real / fake samples 
    gradients=K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
    return(K.mean(gradient_penalty))  

def transform(X):
    return(feature_extractor.predict(X)) 


'wassertein generator'
def build_generator():
    

    model = Sequential()

    model.add(Dense(1000, input_dim=z_size+label_size))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))

    
    model.add(Dense(500))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(100))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
#    model.add(Dense(25,activation='relu'))
    model.add(Dense(50))
    model.add(BatchNormalization(momentum=0.8))
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

#def build_generator():
#    
#
#    model = Sequential()
#
#    model.add(Dense(128 * ret_data.shape[1] , activation="relu", input_dim=z_size+label_size))
#    model.add(Reshape((ret_data.shape[1] ,128)))
#    model.add(Conv1D(64, kernel_size=4, padding="same"))
#    model.add(BatchNormalization(momentum=0.8))
#    model.add(Activation("elu"))
#    model.add(Conv1D(32, kernel_size=4, padding="same"))
#    model.add(BatchNormalization(momentum=0.8))
#    model.add(Activation("elu"))
#    model.add(Conv1D(1, kernel_size=4, padding="same"))
#    model.add(Activation("linear"))
#    
#    model.summary()
#    
#    noise = Input(shape=(z_size,))
#    label = Input(shape=(label_size,))
#    
#    model_input = concatenate([noise, label])
#    output = model(model_input)
#
#    return Model([noise, label], output)

    
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
    model.add(Dense(1,activation='linear')) 
    model.summary()
    
    X_input=Input(shape=(ret_data.shape[1],))
    label = Input(shape=(label_size,))

    model_input = concatenate([X_input, label])

    validity = model(model_input)

    return Model([X_input, label], validity)
    

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


     
def train(X,epoch):
    num_train = X.shape[0]
    start = label_w
#    start=300

    E_disc=[]
    E_gene=[]

    valid = -np.ones((batch_size, 1))
    fake =  np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty

    for e in range(epoch+1):
#        label_w=np.random.randint(0,1000)
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
            
            noise =np.random.normal(0, 0.1, size=(batch_size, z_size))
            d_loss= discr_model.train_on_batch([real_batch, noise, label_batch], [valid, fake, dummy])

        g_loss = generator_model.train_on_batch([noise, label_batch], valid)
        start += batch_size
        
        E_disc.append(d_loss[-1])
        E_gene.append(g_loss)
        
        if start > num_train - batch_size:
            start = label_w
        if e % 10 == 0:
            print(d_loss[-1],d_loss[0],d_loss[1])
            print(e)

            
    E_disc=np.array(E_disc)
    E_gene=np.array(E_gene)
    plt.figure()
    plt.plot(E_gene,label='gene')
    plt.plot(E_disc,label='disc')
    plt.legend()
    
#        if e% epoch == 0:
#        if e==epoch:
#            R_e,R_real=generate_serie()
#            plt.plot(R_e,label='epoch'+str(e))
#    plt.plot(R_real,label='real')
#    plt.legend()
            
  
    


'model de Wassertein'
'1)def du gene + discr'
generator=build_generator()
discriminator=build_discriminator()

'2) discr model'
# Freeze generator's layers while training critic
generator.trainable = False
#
real_data = Input(shape=(ret_data.shape[1],))
noise = Input(shape=(z_size,))
label = Input(shape=(label_size,))
fake_data =generator([noise,label])

#
fake = discriminator([fake_data,label])
valid = discriminator([real_data,label])
#
# Construct weighted average between real and fake data
interpolated_data = RandomWeightedAverage()([real_data, fake_data])
# Determine validity of weighted sample
validity_interpolated =discriminator([interpolated_data,label])
#
# Use Python partial to provide loss function with additional
# 'averaged_samples' argument
partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=interpolated_data)
partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

discr_model = Model(inputs=[real_data, noise, label], outputs=[valid, fake, validity_interpolated])
discr_model.compile(loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss], optimizer=optimizer, loss_weights=[1, 1, 10])

'3) gene model'

discriminator.trainable = False
generator.trainable = True

noise_gen = Input(shape=(z_size, ))
label = Input(shape=(label_size,))
noise_data = generator([noise_gen,label])
valid = discriminator([noise_data,label])

# Defines generator model
generator_model = Model([noise_gen, label], valid)
generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)        


 

def generate_serie():
    R=[0]*label_w
#    noise_t =np.random.normal(0, 1, size=(n_gene,z_size))
    for k in range(label_w,len(ret_data)):
        
        label_t=ret_data[k-label_w:k]
        label_t=label_t.reshape(1,len(label_t))
    ###
        n_gene=1
        noise_t =np.random.normal(0, 0.1, size=(n_gene,z_size))
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

#    plt.figure()
#    plt.plot(R,label='p='+str(label_w))
#    plt.plot(R_real,label='real')
#    plt.xlabel('time')
#    plt.legend()

    
#    return(data_o_gene,data_o)
    return(R,R_real)

#plt.plot(ret_data,label='real')
#plt.legend()    
#    
'training'
epoch=100
train(ret_data,epoch)


R,R_real=generate_serie()
#plt.plot(ret_data)

def get_norm(data):
    mu=data.mean()
    sig=data.std()
    return((data-mu)/sig)
    
R=R[label_w:]
R_n=get_norm(R)
R_real_n=get_norm(R_real)

R_n.index=R_real_n.index

plt.figure()
plt.plot(R_n)
plt.plot(R_real_n)


def get_return(Y):
    R = Y.pct_change()[1:]
    return(R)


ret_real=get_return(R_real_n+3)
ret_gene=get_return(R_n+3)


ret_real.columns=['real']
r=ret_real['real']

ret_gene.columns=['gene']
r_g=ret_gene['gene']

#r_g=r_g[label_w:]

'QQplot'
r_s=np.sort(r)
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
aff_autocorr(250,r,'real')
aff_autocorr(250,r_g,'fake')
plt.legend()


