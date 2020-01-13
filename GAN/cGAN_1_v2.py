# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:22:45 2019

@author: rochej
"""

'CGAN paper 1 avec scaling scikilearn'
'CGAN continuous label: p last days'

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
z_size=200
n_critic = 4
label_w=10

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
batch_size=250


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

    model.add(Dense(200, input_dim=z_size+label_size))
#    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))

    
    model.add(Dense(100))
#    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(50))
#    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.5))
    
#    model.add(Dense(25,activation='relu'))
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

    model.add(Dense(100, input_dim=ret_data.shape[1]+label_size))
    model.add(LeakyReLU(alpha=0.5))
    
    model.add(Dense(50))
    model.add(LeakyReLU(alpha=0.5))
#    model.add(Dropout(0.4))
    
    model.add(Dense(10))
    model.add(LeakyReLU(alpha=0.5))
    
#    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid')) #cgan classique:output is a proba
    model.summary()
    
    X_input=Input(shape=(ret_data.shape[1],))
    label = Input(shape=(label_size,))

    model_input = concatenate([X_input, label])

    validity = model(model_input)

    return Model([X_input, label], validity)
    

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
        
    
       
     
def train(X,epoch):
    num_train = X.shape[0]
    start = label_w
#    start=300

    E_disc=[]
    E_gene=[]

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

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
            
            noise =np.random.normal(0, 0.01, size=(batch_size, z_size))
            
            d_loss_real = discriminator.train_on_batch([real_batch, label_batch], valid)
            gen_imgs = generator.predict([noise, label_batch])
            d_loss_fake = discriminator.train_on_batch([gen_imgs, label_batch], fake)
#            gradients = K.gradients(d_loss_fake, discriminator.input)

##            
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            

##            
        g_loss = combined.train_on_batch([noise, label_batch], valid)

        start += batch_size
        
        E_disc.append(d_loss)
        E_gene.append(g_loss)
        
        if start > num_train - batch_size:
            start = label_w
        if e % 10 == 0:
            print('d_loss_tot=',d_loss)
            print('fake=',d_loss_fake)
            print('real=',d_loss_real)
            print('g_loss=',g_loss)
            print(e)
            print(K.eval(discriminator.optimizer.lr))
            
#    E_disc=np.array(E_disc)
#    E_gene=np.array(E_gene)
#    plt.figure()
#    plt.plot(E_gene,label='gene')
#    plt.plot(E_disc,label='disc')
    
#        if e% epoch == 0:
#        if e==epoch:
#            R_e,R_real=generate_serie()
#            plt.plot(R_e,label='epoch'+str(e))
#    plt.plot(R_real,label='real')
#    plt.legend()
            
  
    
    

'modele simple'
discriminator =build_discriminator()
#discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer, metrics=['accuracy'])
discriminator.compile(loss=['binary_crossentropy'], optimizer=optimizer)
generator = build_generator()


generator.save('gene_spx.h5')
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

def generate_serie():
#    R=[0]*label_w
    R=[]
#    noise_t =np.random.normal(0, 1, size=(n_gene,z_size))
    for k in range(label_w,len(ret_data)):
        
        label_t=ret_data[k-label_w:k]
        label_t=label_t.reshape(1,len(label_t))
    ###
        n_gene=1
        noise_t =np.random.normal(0, 0.0009, size=(n_gene,z_size))
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
    R.index=R_real.index

    plt.figure()
    plt.plot(R,label='sig=0.01')
    plt.plot(R_real,label='real')
    plt.xlabel('time')
    plt.legend()

    
#    return(data_o_gene,data_o)
    return(R,R_real)

#plt.plot(ret_data,label='real')
#plt.legend()    
#    
'training'
epoch=1000
train(ret_data,epoch)

R,R_real=generate_serie()
#plt.plot(ret_data)




'saving model'
#loaded_gene=tf.keras.models.load_model('gene.h5')
#
#noise_t =np.random.uniform(0, 0.01, size=(1,z_size))
#label_t=ret_data[500-label_w:500]
#label_t=label_t.reshape(1,len(label_t))
#print(loaded_gene.predict([noise_t,label_t]))





    



'initial scale'
a=D_sp.min()
b=D_sp.max()

def inv_transform(x):
    return(x * (b - a) + a)

R_ini = R * (b - a) + a
R_real_ini= R_real * (b - a) + a

#plt.plot(R_ini)
#plt.plot(R_real_ini)

#'return'
#ret_r=get_return(pd.DataFrame(scaler.inverse_transform(R_real)))
#ret_g=get_return(pd.DataFrame(scaler.inverse_transform(R)))
##
#plt.plot(ret_g,alpha=0.7,label='fake return')
#plt.plot(ret_r,alpha=0.7,label='real return')
#plt.xlabel('time')
#
#plt.legend()


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
    
def get_return(Y):
    R = Y.pct_change()[1:]
    return(R)
    
ret_real=get_return(R_real_ini)
ret_gene=get_return(R_ini)

    

ret_real.columns=['real']
r=ret_real['real']

ret_gene.columns=['gene']
r_g=ret_gene['gene']

#x_true=np.square(np.array(get_return(D_sp).iloc[:1000]).reshape(len(get_return(R).iloc[:1000]),))
#
x = np.square(np.array(get_return(R_ini).iloc[:1000]).reshape(len(get_return(R).iloc[:1000]),))
y = np.square(np.array(get_return(R_real_ini).iloc[:1000]).reshape(len(get_return(R).iloc[:1000]),))

x = np.array(get_return(R_ini).iloc[:1000]).reshape(len(get_return(R).iloc[:1000]),)
y = np.array(get_return(R_real_ini).iloc[:1000]).reshape(len(get_return(R).iloc[:1000]),)

plt.figure()
aff_autocorr(40,pd.Series(x),'real')
aff_autocorr(40,pd.Series(y),'fake')
plt.legend()




#plt.acorr(x, maxlags=50,label='fake',color='blue',alpha=0.5)
#plt.acorr(y,maxlags=50,label='real',color='red',alpha=0.5)
#plt.legend()
#
#x2 = np.array(R.iloc[:1000]).reshape(len(R.iloc[:1000]),)
#y2=np.array(R_real.iloc[:1000]).reshape(len(R.iloc[:1000]),)
#plt.acorr(x2, maxlags=50,label='fake',color='blue',alpha=0.5)
#plt.acorr(y2,maxlags=50,label='real',color='red',alpha=0.5)
#plt.legend()
#
#y = np.array(get_return(R_real).iloc[:1000]).reshape(len(get_return(R).iloc[:1000]),)
#plt.acorr(x, maxlags=50,label='fake',color='blue',alpha=0.5)
#plt.acorr(y,maxlags=50,label='real',color='red',alpha=0.5)
#plt.legend()
#
#
#aff_autocorr(40,r.iloc[:1000],'real')
#aff_autocorr(40,r_g.iloc[:1000],'fake')
#plt.legend()
##
#plot_acf(r.iloc[:1000], lags=50)
#plot_acf(r_g.iloc[:1000], lags=50)
#plot_acf(R_real.iloc[:1000], lags=100)
#plot_acf(R.iloc[:1000], lags=100)
#pyplot.show()
#
'QQ plot'
#r_g=r_g[label_w:]
#r_s=np.sort(r)
plt.figure()
plt.scatter(np.sort(r), np.sort(r_g))
x=np.array([-1,1])
y=np.array([-1,1])
plt.plot(x,y)
plt.xlabel('reel')
plt.ylabel('gene')
#
'histo'
#F=pd.concat([get_return(R).iloc[300:1000],get_return(R_real).iloc[300:1000]],axis=1)
#F.hist(normed=True,bins=80,alpha = 0.5,axis=1)
#get_return(R).iloc[300:1000].hist(normed=True,bins=80,alpha = 0.5)
#get_return(R_real).iloc[300:1000].hist(normed=True,bins=80,alpha = 0.5)



#x = np.array(get_return(R)).reshape(len(get_return(R)),)
#y = np.array(get_return(R_real)).reshape(len(get_return(R_real)),)
#bins = np.linspace(-0.1, 0.1, 50)
#
#plt.hist([x, y], bins, label=['fake', 'real'])
#plt.legend(loc='upper right')
#plt.show()

'kolmo test'
#ret_real=get_return(R_real).iloc[3900:400]
#ret_gene=get_return(R).iloc[3900:4300]
#xt=np.array(ret_real).reshape(len(ret_real),)
#yt=np.array(ret_gene).reshape(len(ret_real),)
#print(stats.ks_2samp(xt, yt))


    
def simulate_data(n_sim,epoch):
    D=pd.DataFrame()
    for k in range(n_sim):
        
#        train(ret_data,epoch)
        data=generate_serie()
        data_gene=data[0]
        D=pd.concat([D,data_gene],axis=1)
        data_reel=data[1]
        print(k)
        plt.plot(data_gene,label='fake')
#    plt.plot(data_reel,label='reel')
    plt.legend()
    D=pd.DataFrame(D)
    return(D)

#simulate_data(2,500)

#plt.plot(ret_data,label='real')
#plt.legend()    
    
'get mean generated series'
#D=simulate_data(7,500)  
#D.to_csv('D')
#D_ini=inv_transform(D)
#D_ini.columns=[str(i) for i in range(7)]
#R_tot=get_return(D_ini)
#R_s=pd.DataFrame(np.abs(R_tot))
#R_tot=R_s
#A=pd.DataFrame()
#for i in range(7):
#    s=pd.Series(R_tot[str(i)])
#    A=pd.concat([A,aff_autocorr(200,s)],axis=1)
#m_autocorr=A.mean(axis=1)
#
#plt.plot(m_autocorr,label='fake')
#plt.plot(aff_autocorr(200,np.abs(get_return(D_sp))),label='real')
#plt.xlabel('lag')
#plt.ylabel('ACF')
#plt.legend()
#    
#M_tot=R_tot.mean(axis=1)
#
#aff_autocorr(300,M_tot.iloc[:1000],'real')
#aff_autocorr(300,get_return(D_sp).iloc[:1000],'fake')
#plt.legend()


    
def gene_from_loaded(data):
    loaded_gene=tf.keras.models.load_model('gene_spx.h5')
    R=[]
#    noise_t =np.random.normal(0, 1, size=(n_gene,z_size))
    for k in range(label_w,len(data)):
        
        label_t=data[k-label_w:k]
        label_t=label_t.reshape(1,len(label_t))
    ###
        n_gene=1
        noise_t =np.random.normal(0, 0.001, size=(n_gene,z_size))
    #    noise_t =np.random.normal(0, 0.01, size=(n_gene,z_size))
    ##noise_t=np.array([1 for i in range(100)]).reshape(1,100)
    ##noise_t=ret_real.iloc[:100]
    ##noise_t=np.array(noise_t).reshape(1,len(noise_t))
    #
        ret_gene=loaded_gene.predict([noise_t,label_t])
    #    ret_gene=pd.DataFrame(ret_gene).T
        R.append(float(ret_gene))
        
    R=pd.DataFrame(R) #R: output du generateur
    R_real=pd.DataFrame(data[label_w:]) #rett_data: zscore des return du SPX
    
    plt.plot(R,label='gene')
    plt.plot(R_real,label='reel')
    plt.legend()
    return(R,R_real)
    

    
#ret_data = scaler.fit_transform(df_sp.reshape(-1, 1))
#data=ret_data
#gene_from_loaded(data)
    
