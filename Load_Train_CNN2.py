import matplotlib.pyplot as plt
import random
import pickle
import numpy as np
from os import path
import os.path
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd
import time
from os import listdir
from shutil import copyfile
import pickle

def create_seed():
    return random.randint(1, 1000000)



def find_min_num(directory):
    print('Balancing the TRAINING data with the smallest class')
    numbers = []
    
    for path, folders, files in os.walk(directory):        
      
      for folder in folders:
        
        dolphin_X = []         
        
        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'    
        with open(training_file) as fp:
          line = fp.readline()

          while line:
            file_name = line.strip()
            file_name = file_name[:file_name.find('.wav')]            
            dolphin_X.extend(pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_augmented_img.pkl', "rb" )))
            dolphin_num = np.asarray(dolphin_X)            
            line = fp.readline() 

        number = int(len(dolphin_X))
        numbers.append(number)
        #print('number of samples: ', number)

    numbers = np.asarray(numbers)
    numbers.sort()#min() was not working idk why!!
    minimum_sample = numbers[:1]
    
    minimum = int(numbers[:1])
    
    
    
    print()
    print('Training dataset will be balanced to {} samples per species'.format(minimum))
     

    return minimum

def find_min_num_val(directory):
    print('Balancing the VALIDATION data with the smallest class')
    numbers = []
    
    for path, folders, files in os.walk(directory):        
      
      for folder in folders:
        
        dolphin_X = [] 
        
        

        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
    
        with open(training_file) as fp:
          line = fp.readline()

          while line:
            file_name = line.strip()
            file_name = file_name[:file_name.find('.wav')]            
            dolphin_X.extend(pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_img.pkl', "rb" )))
            dolphin_num = np.asarray(dolphin_X)            
            line = fp.readline() 

        number = int(len(dolphin_X))
        numbers.append(number)
        #print('number of samples: ', number)

    numbers = np.asarray(numbers)
    numbers.sort()
    minimum_sample = numbers[:1]
    
    minimum_val = int(numbers[:1])
    
    
    
    print()
    print('Training dataset will be balanced to {} samples per species'.format(minimum_val))
     

    return minimum_val           
            

def load_validation_images_CNN2(species_dir_val):
    print('Loading VALIDATION data for the second Convolutional Neural Network (CNN-2): Species identification')
    X_val = []
    Y_val = []
    
    label = 0
    minimum = find_min_num_val(species_dir_val)
    

    
    print()
    
    print('Dolphin data files by species:')
    
    
    for path, folders, files in os.walk(species_dir_val):        
      
      for folder in folders:
        print()
        dolphin_X = [] 
        print(' %s ' % os.path.basename(folder))

        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
    
        with open(training_file) as fp:
          line = fp.readline()

          while line:

            file_name = line.strip()
            
            
            print ('{}'.format(file_name))
            file_name = file_name[:file_name.find('.wav')]
            
            dolphin_X.extend(pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_img.pkl', "rb" )))
            
            line = fp.readline() 
                                 
        dolphin_X = np.asarray(dolphin_X)
        print()
        print('Unbalanced %s data shape: ' % os.path.basename(folder), dolphin_X.shape)
        dolphin_X = dolphin_X[np.random.choice(dolphin_X.shape[0], minimum, replace=True)]
        print('Balanced %s data shape: ' % os.path.basename(folder), dolphin_X.shape)
        X_val.extend(dolphin_X)
        Y = np.full(len(dolphin_X), fill_value=label)
        Y_val.extend(Y)
        label = label + 1   

    X = np.asarray(X_val)
    Y = np.asarray(Y_val)
    Y = to_categorical(Y)

    
    
  

    return X, Y
 

   
def load_training_images_CNN2(species_dir):
    print('Loading data for the second Convolutional Neural Network (CNN-2): Species identification')
    X_train = []
    Y_train = []
    
    label = 0
    minimum = find_min_num(species_dir)
    

    
    print()
    
    print('Dolphin data files by species:')
    
    
    for path, folders, files in os.walk(species_dir):        
      
      for folder in folders:
        print()
        dolphin_X = [] 
        print(' %s ' % os.path.basename(folder))

        #Get training file:
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
    
        with open(training_file) as fp:
          line = fp.readline()

          while line:

            file_name = line.strip()
            
            
            print ('{}'.format(file_name))
            file_name = file_name[:file_name.find('.wav')]
            
            dolphin_X.extend(pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_augmented_img.pkl', "rb" )))
            
            line = fp.readline() 
                                 
        dolphin_X = np.asarray(dolphin_X)
        print()
        print('Unbalanced %s data shape: ' % os.path.basename(folder), dolphin_X.shape)
        dolphin_X = dolphin_X[np.random.choice(dolphin_X.shape[0], minimum, replace=True)]
        print('Balanced %s data shape: ' % os.path.basename(folder), dolphin_X.shape)
        X_train.extend(dolphin_X)
        Y = np.full(len(dolphin_X), fill_value=label)
        Y_train.extend(Y)
        label = label + 1   

    X = np.asarray(X_train)
    Y = np.asarray(Y_train)
    Y = to_categorical(Y)

    image_size = int(X.shape[1])
    
  

    return X, Y, image_size



from sklearn.utils import compute_class_weight



def train_model_CNN2(number_iterations, species_dir,species_dir_val,network_CNN2, plot_confusion_matrix,decision_val,location_models,prediction_folder):
    
    print()
    
    print('...............................................')
    print ('Processing data for CNN-2...')
    X_train, Y_train, image_size = load_training_images_CNN2(species_dir)
    if decision_val == 0:
      X_val, Y_val = load_validation_images_CNN2(species_dir_val)

    if decision_val == 1:
      seed = create_seed()
      X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=seed, shuffle = True)
     
    
    list_classes = []
    number = 0
    for path, folders, files in os.walk(species_dir):    
    	for folder in folders:        
        	list_classes.append(number)        
        	number = number + 1
    class_weights={key: 1. for key in list_classes}
    class_names = os.listdir(species_dir)
    


    
    print ('Processing done.')
    print('...............................................')
    print('...............................................')
    
    print('Summary CNN-2:')
    print()
    
    print ('Shape of X', X_train.shape)
    print ('Shape of Y', Y_train.shape)
    
    seed = create_seed()
        
    for experiment_id in range(0,number_iterations):

        print('Iteration {} starting...'.format(experiment_id))

        print ('experiment_id: {}'.format(experiment_id))
        
        
                                                           
        
        
        print ('X_train:',X_train.shape)
        print ('Y_train:',Y_train.shape)
        print ()
        print ('X_val:',X_val.shape)
        print ('Y_val:',Y_val.shape)

        
        filepath= location_models + '/weights_{}_CNN2.hdf5'.format(seed)
        
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
        
        
        model = network_CNN2(image_size, species_dir)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        
        model.summary()
        
        start = time.time()
        
        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                  batch_size=32,
                  epochs=20,
                  verbose=2, 
                  callbacks=[checkpoint], 
                  class_weight=class_weights)
        end = time.time()
        
        
        model.load_weights(location_models + '/weights_{}_CNN2.hdf5'.format(seed))
        
        
        
        train_acc = accuracy_score(np.argmax(model.predict(X_train), axis=-1), np.argmax(Y_train,1))
        
        print (train_acc)
        
        
        val_acc = accuracy_score(np.argmax(model.predict(X_val), axis=-1), np.argmax(Y_val,1))
        
        print (val_acc)
        
        
        
        cnf_matrix = confusion_matrix(np.argmax(Y_val,1), np.argmax(model.predict(X_val), axis=-1))
        np.set_printoptions(precision=2)

        
        plt.figure()
        
        print ()
        print ('Plotting performance on validation data.')
        
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        
        plt.savefig(location_models +'/Confusion_Matrix_Model_{}.png'.format(seed))
        
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        
        specificity = TN/(TN+FP)
        sensitivity = TP/(FN+TP)

        FPR = 1 - specificity
        FNR = 1 - sensitivity
        
        performance = []
        performance.append(train_acc)
        performance.append(val_acc)
        performance.append(end-start)

        
        np.savetxt(location_models + '/train_test_performance_{}_CNN2.txt'.format(seed), np.asarray(performance), fmt='%f') 
        
        
        with open(location_models + '/history_{}_CNN2.txt'.format(seed), 'wb') as file_out:
                pickle.dump(history.history, file_out)

        print('Iteration {} ended...'.format(experiment_id))
        print('Results saved to:')
        print(location_models + 'train_test_performance_{}_CNN2.txt'.format(seed))
        
        print('-------------------')
        time.sleep(1)