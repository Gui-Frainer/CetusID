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

#Confusion Matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    
    
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
#Preparing validation dataset:

def create_seed():
    return random.randint(1, 1000000)

#CNN1: dolphin x non-dolphin
def load_validation_images_CNN1(species_dir_val, soundscape_dir_val):
    print('Loading validation data for the first Convolutional Neural Network (CNN-1): Dolphin detection')

    dolphin_X = []
    nondolphin_X = []

    
    print()
    
    print('Dolphin data files by species:')
    
    
    for path, folders, files in os.walk(species_dir_val):        
       
      for folder in folders:
        print()
        
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
    print('...............................................')
    
    print('Non-dolphin data files by place:')
    

    for path, folders, files in os.walk(soundscape_dir_val):
       
        
       for folder in folders:
        print()
        
        print(' %s ' % os.path.basename(folder))

        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
    
        with open(training_file) as fp:
          line = fp.readline()

          while line:

            file_name = line.strip()
            
            
            print ('{}'.format(file_name))
            file_name = file_name[:file_name.find('.wav')]
            
            nondolphin_X.extend(pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_img.pkl', "rb" )))
                         


            
            line = fp.readline()
            


    nondolphin_X = np.asarray(nondolphin_X)
    print()
    print('...............................................')

     #Balancing the data:
    sample_amount = dolphin_X.shape[0]
    nondolphin_X = nondolphin_X[np.random.choice(nondolphin_X.shape[0], sample_amount, replace=True)]
    
    
    
    
    
    print('...............................................')
    print('...............................................')
    print('Summary:')
    print()
    
    
    print('Dolphin data:', dolphin_X.shape)
    print('Nondolphin data:', nondolphin_X.shape)
    print()
    #print('Nondolphin data (unbalanced):',nondolphin_XX.shape )

    
    return dolphin_X, nondolphin_X

def prepare_X_and_Y_CNN1_val(dolphin_X, nondolphin_X):

    Y_dolphin = np.ones(len(dolphin_X))
    Y_nondolphin = np.zeros(len(nondolphin_X))
    X_val = np.concatenate([dolphin_X, nondolphin_X])
    del dolphin_X, nondolphin_X
    Y_val = np.concatenate([Y_dolphin, Y_nondolphin])
    del Y_dolphin, Y_nondolphin
    Y_val = to_categorical(Y_val)

    return X_val, Y_val

def create_validation(species_dir_val, soundscape_dir_val):#First part, sorting the images
    
    training_files = []
    dolphin_X, nondolphin_X = load_validation_images_CNN1(species_dir_val, soundscape_dir_val)
    
    print()
    
    print('...............................................')
    print ('Processing data for CNN-1...')
    X_val, Y_val = prepare_X_and_Y_CNN1_val(dolphin_X, nondolphin_X)
    del dolphin_X, nondolphin_X
    print ('Processing done.')
    print('...............................................')
    print('...............................................')
    
    

    return X_val, Y_val
    

#CNN1: dolphin x non-dolphin
def load_training_images_CNN1(species_dir, soundscape_dir):
    print('Loading data for the first Convolutional Neural Network (CNN-1): Dolphin detection')

    dolphin_X = []
    nondolphin_X = []

    
    print()
    
    print('Dolphin data files by species:')
    
    
    for path, folders, files in os.walk(species_dir):        
       
      for folder in folders:
        print()
        
        print(' %s ' % os.path.basename(folder))

        
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
    print('...............................................')
    
    print('Non-dolphin data files by place:')
    

    for path, folders, files in os.walk(soundscape_dir):
       
        
       for folder in folders:
        print()
        
        print(' %s ' % os.path.basename(folder))

        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
    
        with open(training_file) as fp:
          line = fp.readline()

          while line:

            file_name = line.strip()
            
            
            print ('{}'.format(file_name))
            file_name = file_name[:file_name.find('.wav')]
            
            nondolphin_X.extend(pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_augmented_img.pkl', "rb" )))

            
            line = fp.readline()
            


    
    
    nondolphin_X = np.asarray(nondolphin_X)#to balance with the soundscape data
    print()
    print('...............................................')

    #Balancing the data:
    sample_amount = dolphin_X.shape[0]
    nondolphin_X = nondolphin_X[np.random.choice(nondolphin_X.shape[0], sample_amount, replace=True)]
    
    
    
    
    
    
    dolphin_X = dolphin_X[np.random.choice(dolphin_X.shape[0], sample_amount, replace=True)]
    image_size = int(dolphin_X.shape[1])
    
    print ('Data loaded. ')
    
    print('...............................................')
    print('...............................................')
    print('Summary:')
    print()
    
    
    print('Dolphin data:', dolphin_X.shape)
    print('Nondolphin data:', nondolphin_X.shape)
    print()
    

    
    return dolphin_X, nondolphin_X, image_size

def prepare_X_and_Y_CNN1(dolphin_X, nondolphin_X):

    Y_dolphin = np.ones(len(dolphin_X))
    Y_nondolphin = np.zeros(len(nondolphin_X))
    X = np.concatenate([dolphin_X, nondolphin_X])
    del dolphin_X, nondolphin_X
    Y = np.concatenate([Y_dolphin, Y_nondolphin])
    del Y_dolphin, Y_nondolphin
    Y = to_categorical(Y)

    return X, Y
    



def train_model_CNN1(number_iterations, species_dir, soundscape_dir,species_dir_val,soundscape_dir_val,network_CNN1, plot_confusion_matrix, decision_val,location_models,prediction_folder):
    
    training_files = []
    dolphin_X, nondolphin_X, image_size = load_training_images_CNN1(species_dir, soundscape_dir)
    
    print()
    
    print('...............................................')
    print ('Processing data for CNN-1...')
    X_train, Y_train = prepare_X_and_Y_CNN1(dolphin_X, nondolphin_X)
    del dolphin_X, nondolphin_X
    print ('Processing done.')
    print('...............................................')
    print('...............................................')
    
    print('Summary CNN-1:')
    print()
    
    print ('Shape of X', X_train.shape)
    print ('Shape of Y', Y_train.shape)
    
    seed = create_seed()
        
    for experiment_id in range(0,number_iterations):

        print('Iteration {} starting...'.format(experiment_id))

        print ('experiment_id: {}'.format(experiment_id))
        
        if decision_val == 0:
          X_val, Y_val = create_validation(species_dir_val, soundscape_dir_val)
        if decision_val == 1:
          X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.20, random_state=seed, shuffle = True)
        
        print ('X_train:',X_train.shape)
        print ('Y_train:',Y_train.shape)
        print ()
        print ('X_val:',X_val.shape)
        print ('Y_val:',Y_val.shape)

        
        filepath= location_models + '/weights_{}_CNN1.hdf5'.format(seed)
        
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',verbose=1, save_best_only=True, mode='max')
        
        
        model = network_CNN1(image_size)
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        
        model.summary()
        
        start = time.time()

        history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), 
                  batch_size=32,
                  epochs=20,
                  verbose=2, 
                  callbacks=[checkpoint], 
                  class_weight={0:1.,1:1.})
        end = time.time()
        
        
        model.load_weights(location_models + '/weights_{}_CNN1.hdf5'.format(seed))
        
        
        train_acc = accuracy_score(np.argmax(model.predict(X_train), axis=-1), np.argmax(Y_train,1))
        print (train_acc)
        
        val_acc = accuracy_score(np.argmax(model.predict(X_val), axis=-1), np.argmax(Y_val,1))
        print (val_acc)
        
        
        cnf_matrix = confusion_matrix(np.argmax(Y_val,1), np.argmax(model.predict(X_val), axis=-1))
        np.set_printoptions(precision=2)

        
        plt.figure()
        class_names=['Soundscape','Dolphin']

        print ()
        print ('Plotting performance on validation data.')
        
        plt.figure()
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                              title='Normalized confusion matrix')

        
        plt.savefig(location_models +'/Confusion_Matrix_Model_{}.png'.format(seed))
        
        TN = cnf_matrix[0][0]
        FP = cnf_matrix[0][1]
        FN = cnf_matrix[1][0]
        TP = cnf_matrix[1][1]

        specificity = TN/(TN+FP)
        sensitivity = TP/(FN+TP)

        FPR = 1 - specificity
        FNR = 1 - sensitivity
        
        performance = []
        performance.append(train_acc)
        performance.append(val_acc)
        performance.append(end-start)

        
        np.savetxt(location_models +'/train_test_performance_{}_CNN1.txt'.format(seed), np.asarray(performance), fmt='%f')

        
        with open(location_models +'/history_{}_CNN1.txt'.format(seed), 'wb') as file_out:
                pickle.dump(history.history, file_out)

        print('Iteration {} ended...'.format(experiment_id))
        print('Results saved to:')
        print('{}/train_test_performance_{}_CNN1.txt'.format(location_models,seed))
        
        print('-------------------')
        time.sleep(1)