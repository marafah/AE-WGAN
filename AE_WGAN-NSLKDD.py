from __future__ import print_function, division

import time

from keras.layers import Input, Dense, Reshape, Flatten, concatenate, BatchNormalization
from keras.models import Sequential, Model
from keras import optimizers
import keras.backend as K

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Preprocessing as p
import Generative_scores as gs

import os

import warnings
warnings.filterwarnings("ignore")

start_time_training = time.time()

def AE_WGAN():

    class AE_WGAN():
        def __init__(self):
            self.img_rows = 4
            self.img_cols = 6
            self.channels = 1
            self.img_shape = (self.img_rows, self.img_cols, self.channels)
            self.latent_dim = 32
            
            self.sum_time_generating = 0
    
            optimizer = optimizers.Adamax(.00009)
    
            #Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss=[self.wasserstein_loss],
                optimizer=optimizer,
                metrics=['accuracy'])
    
            #Build the generator
            self.generator = self.build_generator()
    
            #Build the encoder
            self.encoder = self.build_encoder()
    
            #The part of the AE_WGAN that trains the discriminator and encoder
            self.discriminator.trainable = False
    
            #Generate image from sampled noise
            z = Input(shape=(self.latent_dim, ))
            img_ = self.generator(z)
    
            #Encode image
            img = Input(shape=self.img_shape)
            z_ = self.encoder(img)
    
            #Latent -> img is fake, and img -> latent is valid
            fake = self.discriminator([z, img_])
            valid = self.discriminator([z_, img])
    
            #Set up and compile the combined model
            #Trains generator to fool the discriminator
            self.AE_WGAN_generator = Model([z, img], [fake, valid])
            self.AE_WGAN_generator.compile(loss=[self.wasserstein_loss, self.wasserstein_loss],
                optimizer=optimizer)
            
        def wasserstein_loss(self, y_true, y_pred):
            return K.mean(y_true * y_pred)
    
        def build_encoder(self):
            model = Sequential()
    
            model.add(Flatten(input_shape=self.img_shape))
            model.add(Dense(20, activation='relu'))
            model.add(Dense(self.latent_dim))
    
            model.summary()
    
            img = Input(shape=self.img_shape)
            z = model(img)
    
            return Model(img, z)
    
        def build_generator(self):
            model = Sequential()
    
            model.add(Dense(20, input_dim=self.latent_dim, activation='relu'))
            model.add(Dense(40, activation='relu'))
            model.add(Dense(80, activation='relu'))
            model.add(Dense(120, activation='relu'))
            model.add(Dense(160, activation='relu'))
            
            model.add(Dense(np.prod(self.img_shape), activation='relu'))
            model.add(Reshape(self.img_shape))
    
            model.summary()
    
            z = Input(shape=(self.latent_dim,))
            gen_img = model(z)
    
            return Model(z, gen_img)
    
        def build_discriminator(self):
            
            model_ = Sequential()
            model_.add(Dense(80, input_dim=self.latent_dim, activation='relu'))
            model_.add(Dense(40, activation='relu'))
            model_.add(Dense(1, activation='relu'))
            model_.summary()
    
            z = Input(shape=(self.latent_dim, ))
            img = Input(shape=self.img_shape)
            d_in = concatenate([z, Flatten()(img)])
    
            model = Dense(80, activation='relu')(d_in)
            model = Dense(40, activation='relu')(model)
            validity = Dense(1, activation="sigmoid")(model)
    
            return Model([z, img], validity)
    
        def train(self, attack_name, epochs, batch_size=32):
            
            epoch_list = []
            loss_D = []
            acc_D = []
            loss_G = []
    
            #Adversarial ground truths
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
    
            for epoch in range(epochs):
    
    
                # ---------------------
                #  Train Discriminator
                # ---------------------
    
                #Sample noise and generate img
                z = np.random.normal(0, 1, size=(batch_size, self.latent_dim))
                imgs_ = self.generator.predict(z)
    
                #Select a random batch of images and encode
                idx = np.random.randint(0, reshaped_X_attack.shape[0], batch_size)
                imgs = reshaped_X_attack[idx]
                z_ = self.encoder.predict(imgs)

                #Train the discriminator (img -> z is valid, z -> img is fake)
                d_loss_real = self.discriminator.train_on_batch([z_, imgs], valid)
                d_loss_fake = self.discriminator.train_on_batch([z, imgs_], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
                # ---------------------
                #  Train Generator
                # ---------------------
    
                #Train the generator (z -> img is valid and img -> z is is invalid)
                g_loss = self.AE_WGAN_generator.train_on_batch([z, imgs], [valid, fake])
    
                #Plot the progress
                print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0]))
                    
                if epoch %10 == 0:        
                    
                    loss_D.append(d_loss[0])
                    acc_D.append(100*d_loss[1])
                    loss_G.append(g_loss[0])
                    
                epoch_list.append(epoch)
                    
                if epoch == 100:
                    print('\n')
                    print('Generating the attack data(',attack_name,')...\n')
                    generated_pics=self.generate_data()
                    
                    len_loss = len(loss_D)
                        
                    len_epochs = len(epoch_list)
                        
                    o = len_epochs/len_loss
                    
                    epoch_list.clear()
                    
                    for _ in range(len(loss_D)):
                        
                        epoch_list.append(o)
    
                        len_loss=len_loss-1
                        
                        if len_loss == 0:
                            
                            break
                        
                        o = len_epochs/len_loss
    
                    break
                    
            plt.figure(figsize=(10,7))
            plt.subplot(2, 1, 1)
            plt.title(attack_name)
            plt.plot(epoch_list, loss_D,'-o', label='Discriminator loss')
            plt.plot(epoch_list, loss_G,'-s', label='Generator loss')
            plt.ylabel('loss')
            plt.legend()
    
            plt.subplot(2, 1, 2)
            plt.plot(epoch_list, acc_D, '-s', label='Discriminator accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epochs')
            plt.legend()
            
            plt.savefig('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\plots\\E_AE_WGAN_selectedAttacks_%s.jpeg' % attack_name, bbox_inches='tight')
            plt.show()
            
            cwd = os.getcwd()
            print(cwd + '\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\plots\\')
            
            return generated_pics
    
        def generate_data(self):
            
            generated_pics = []
    
            #start_time_generating = time.time()
    
            a = int(len(reshaped_X_attack)+20000)

            for x in range(a):
                
                noise = np.random.normal(size=(32, self.latent_dim))
                
                gen_imgs = self.generator.predict(noise)
                    
                generated_pics.append(gen_imgs[0])
                
            #print(self.sum_time_generating)
                
            #self.sum_time_generating = self.sum_time_generating + (time.time() - start_time_generating)
            
            #print(self.sum_time_generating)
    
            return generated_pics
    
    all_attacks = ['buffer_overflow' ,'land', 'rootkit', 'ps', 'loadmodule', 
                   'perl', 'xsnoop', 'udpstorm', 'sqlattack', 'warezmaster',
                   'warezclient', 'snmpgetattack', 'snmpguess', 'processtable',
                   'multihop', 'phf', 'sendmail', 'xterm', 'named', 'ftp_write',
                   'xlock', 'guess_passwd', 'spy', 'worm', 'imap',
                   'ipsweep', 'nmap', 'portsweep', 'satan', 'saint', 'mscan',
                   'pod', 'mailbomb','apache2', 'normal']

    #This is the starting point of the code
    for attack_name in all_attacks:
    
        #Test data splited into normal and attack
        attack = pd.read_csv('.\\Datasets\\NSL-KDD\\Preprocessing_data\\attack names\\%s.csv' % attack_name)
        
        #Spliting the normal data into features and label
        y_attack=attack.iloc[:,[-1]]
        X_attack=attack.drop(y_attack.columns,axis = 1)
        
        #Reshaping the normal and attack data as 4x6 images to be used in AE_WGAN
        reshaped_X_attack = np.asarray(X_attack).reshape(-1, 4, 6, 1)
        
        ae_wgan = AE_WGAN()
        generated_pics = ae_wgan.train(attack_name, epochs=101, batch_size=32)
        
        #Converting the generated images as normal into numpy array
        generated_attack = np.array(generated_pics)
        
        #Reshaping the features from images to tabular format
        generated_attack = generated_attack.reshape(-1,24)
    
        a = np.array(X_attack.columns)
    
        q = pd.DataFrame(generated_attack, columns=a)
        
        for y in range(attack.shape[0]):
            
            q['class'] = attack_name
        
        q.to_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks\\%s.csv' % attack_name, index=False)

    t = int(time.time() - start_time_training) / 60
    print("--- %s minutes ---" % t)
    
    #Saving all the attack in one dataset and then save by each attack category
    df = pd.DataFrame(columns = a)
    
    for x in all_attacks:
        
        generated = pd.read_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks\\%s.csv' % x)
        
        df = pd.concat([df, generated])
    
    df.to_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks\\all_attacks.csv', index=False)
    
    df = pd.read_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks\\all_attacks.csv')
    
    df_dic = p.dictionary(df)
    
    R2L = df_dic[df_dic['class'] == 'R2L']
    R2L.to_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks 5 categories\\R2L.csv', index=False)
    
    U2R = df_dic[df_dic['class'] == 'U2R']
    U2R.to_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks 5 categories\\U2R.csv', index=False)
    
    Probe = df_dic[df_dic['class'] == 'Probe']
    Probe.to_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks 5 categories\\Probe.csv', index=False)
    
    DoS = df_dic[df_dic['class'] == 'DoS']
    DoS.to_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks 5 categories\\DoS.csv', index=False)
      
    normal = df_dic[df_dic['class'] == 'normal']
    normal.to_csv('.\\Datasets\\NSL-KDD\\generated\\AE_WGAN\\generated attacks 5 categories\\normal.csv', index=False)
    
    # Calculating and saving the cosine similarity
    gs.Cosine_Similarity('AE_WGAN', all_attacks)
    
    # Calculating and saving the Maximum Mean Discrepancy
    print('\nThe Maximum Mean Discrepancy is:')
    gs.Maximum_Mean_Discrepancy('AE_WGAN')
