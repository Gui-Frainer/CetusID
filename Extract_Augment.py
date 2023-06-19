import pandas as pd
import librosa
from scipy import signal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import struct
from scipy.signal import argrelextrema
from scipy.fftpack import rfft, irfft, fftfreq
import time
import librosa.display
import scipy.fftpack
from glob import glob
from scipy import signal
import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas as pd
import struct
from scipy.signal import argrelextrema
from scipy.fftpack import rfft, irfft, fftfreq
import time
import librosa
import librosa.display
import scipy.fftpack
from os import listdir
import random
from shutil import copyfile
import pickle
import os
from skimage import data, color




seed = 42
number_iterations = 1
augmentation_probability = 0.3 ##Change this to 1 when convenient...
augmentation_amount_soundscape = 6
augmentation_amount = 1
jump_seconds_dolphin = 1 #Change this to 0.something when convenient...
jump_seconds_soundscape = 1 #Change this to 0.something when convenient...

#Building training files:


def build_training_files_list(directory):  
    with open(directory + '/Training_Files.txt', "w") as f:
      for x in sorted(os.listdir(directory)):
        if x.endswith(".wav"):
          f.write(x +'\n')

def execute_listing_folders(parent_directory):
    for path, folders, files in os.walk(parent_directory):
      for folder in folders:
        build_training_files_list (os.path.join(path,folder))

#Extract audio:

def read_and_process_timestamps(directory, file_name, sample_rate, sep):       
        timestamps = pd.read_csv(directory+file_name, sep=sep)
        timestamps['Start'] = timestamps['Start'] * sample_rate
        timestamps['End'] = timestamps['End'] * sample_rate
        timestamps.columns = ['Start', 'End',' Duration']
        return timestamps

def extract_audio (librosa_audio, timestamps = None, alpha = 5, jump_seconds=0.4,sample_rate=0, verbose=0):

    alpha_converted = alpha * sample_rate
    extracted = []
    audio_length = librosa.get_duration(librosa_audio, sample_rate)

    
    for index, row in timestamps.iterrows(): 
        
        
        

        jump = 0

        while True:
            
            start_position = row['Start'] - (sample_rate) + (jump * jump_seconds * sample_rate)           
            
            
            if start_position < 0:
              start_position = 0
              end_position = start_position + alpha_converted

              
            end_position = start_position + alpha_converted
            jump = jump + 1

            if end_position > audio_length * sample_rate:
                if verbose:
                    print('Breaking.')
                break              

            if verbose:
                print ('start_position',start_position)
                print ('end_position',end_position)
                print ()
            if start_position > row['End'] - sample_rate:
                if verbose:
                    print('Breaking.')
                break
            
            extracted.append(librosa_audio[int(start_position):int(end_position)])

    return np.asarray(extracted)

def execute_audio_extraction_dolphin(audio_directory, audio_file_name, sample_rate,
                            time_to_extract):
    
    print ('Reading audio file (this can take some time)...')
    
    librosa_audio, librosa_sample_rate = librosa.load(audio_directory+audio_file_name, 
                                                  sr=sample_rate)
    
    print ()
    print ('Reading done.')
    
    
    timestamps = read_and_process_timestamps(audio_directory, 
                                   audio_file_name[:audio_file_name.find('.wav')]+ '.csv', 
                                               sample_rate, sep=',')
    
    print ('file: ', audio_file_name)
    extracted = extract_audio(librosa_audio, timestamps, time_to_extract,jump_seconds_dolphin, sample_rate,0)
    
    print()
    print('Extracted audio shape:', extracted.shape)
    
    
    pickle.dump(extracted, open(audio_directory+audio_file_name[:audio_file_name.find('.wav')]+'_audio.pkl', "wb" ))
    
    del librosa_audio
    print ()
    print ('Extracting segments done. Pickle files saved.')
    
    return extracted

def execute_audio_extraction_soundscape(audio_directory, audio_file_name, sample_rate,
                            time_to_extract):
    
    print ('Reading audio file (this can take some time)...')
    
    librosa_audio, librosa_sample_rate = librosa.load(audio_directory+audio_file_name, 
                                                  sr=sample_rate)
    
    print ()
    print ('Reading done.')
    
    
    timestamps = read_and_process_timestamps(audio_directory, 
                                   audio_file_name[:audio_file_name.find('.wav')]+ '.csv', 
                                               sample_rate, sep=',')
    
    print ('file: ', audio_file_name)
    extracted = extract_audio(librosa_audio, timestamps, time_to_extract,jump_seconds_soundscape, sample_rate,0)
    
    print()
    print('Extracted audio shape:', extracted.shape)
    
    
    pickle.dump(extracted, open(audio_directory+audio_file_name[:audio_file_name.find('.wav')]+'_audio.pkl', "wb" ))
    
    del librosa_audio
    print ()
    print ('Extracting segments done. Pickle files saved.')
    
    return extracted


def convert_to_spectrograms (audio, dpi):

    spectrograms = []

    for data in audio:
        S = librosa.stft(data, n_fft=1024)
        fig = plt.figure(figsize=[5, 5], dpi=dpi)
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        plt.subplots_adjust(0,0,1,1,0,0)
        librosa.display.specshow(librosa.amplitude_to_db(abs(S)))
        
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        image_array = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        spectrograms.append(image_array)
        
        plt.close('all')


    spectrograms = np.asarray(spectrograms)
    spectrograms = np.reshape(spectrograms, (spectrograms.shape[0],spectrograms.shape[1],spectrograms.shape[2],3)) 
    spectrograms = spectrograms[:,:,:,:1]
    
    return spectrograms 

def blend(audio_1, audio_2, w_1, w_2):
    augmented = w_1 * audio_1 + w_2 * audio_2
    return augmented



def time_shift(audio, time, sample_rate):

    augmented = np.zeros(len(audio))
    augmented [0:sample_rate*time] = audio[-sample_rate*time:]
    augmented [sample_rate*time:] = audio[:-sample_rate*time]
    return augmented

def augment_data(seed, augmentation_amount, augmentation_probability,
                 extracted, soundscape, sample_rate, alpha):
    

    np.random.seed(seed)
    random.seed(seed)
    
    augmented_data = []
    
    
    for extracted_data in extracted:
        
        
        for i in range (0, augmentation_amount):
        
            
            probability = random.random()
            
            
            
            if probability <= augmentation_probability:

                
                random_soundscape = random.randint(0, len(soundscape)-1)

                
                random_time_point = random.randint(1, alpha-1)

                
                new_data = time_shift(soundscape[random_soundscape], random_time_point, sample_rate)

                
                new_data = blend(extracted_data, new_data, 0.9, 0.1)
                
                
                
                augmented_data.append(new_data)
                
                

    
    return np.asarray(augmented_data)


def execute_augmentation(extracted, 
                                  soundscape, time_to_extract, sample_rate,
                                  augmentation_amount, augmentation_probability, 
                                  seed,
                                  audio_file_name, audio_directory):
    
    print()
    print ('extracted:',extracted.shape)
 
    
    
    extracted_augmented = augment_data(seed, augmentation_amount, 
                                              augmentation_probability, extracted, 
                                              soundscape, sample_rate, 
                                              time_to_extract)
    

    
    print()
    print('extracted_augmented:',extracted_augmented.shape)
    


       
    return extracted_augmented
    
    print()
    print ('Augmenting done. Jpeg files saved to %s folder...' % os.path.basename(audio_directory))

def extract_validation(species_dir_val,soundscape_dir_val,
                            sample_rate, number_seconds_to_extract, augmentation_probability, 
                            augmentation_amount, seed, number_iterations, dpi):
  
     
     
     print('Preparing validation files: ...')        
     execute_listing_folders(species_dir_val)
     execute_listing_folders(soundscape_dir_val)
     print('Training files written.')
     print()
     
     print('Loading soundscape data:')

      

     for path, folders, files in os.walk(soundscape_dir_val):
       for folder in folders:
        print()
        print('----------------------------------')
        print('Extracting audio data from %s' % os.path.basename(folder))
                
        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))
                
                file_name_x = '/'+file_name
                
                soundscape_sound = execute_audio_extraction_soundscape(os.path.join(path,folder),file_name_x, sample_rate, number_seconds_to_extract)
                file_name = file_name[:file_name.find('.wav')]

                soundscape_image = convert_to_spectrograms(soundscape_sound, dpi)
                
       
    
                pickle.dump(soundscape_image, open(os.path.join(path,folder)+'/'+file_name+'_img.pkl', "wb" ))
                

               
                line = fp.readline() 

     

    
     print()
     print('----------------------------------')
     print('----------------------------------')
     print('Loading dolphin data:')
     
     
     for path, folders, files in os.walk(species_dir_val):
       
       for folder in folders:
        print()
        print('----------------------------------')
        print('Extracting audio data of %s' % os.path.basename(folder))

        temp_total_audio = []
        temp_total_augmented = []
        temp_total_image = []

 

        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))

                
                file_name_x = '/'+file_name
                
                dolphin_sound = execute_audio_extraction_dolphin(os.path.join(path,folder),file_name_x, sample_rate, number_seconds_to_extract)
                temp_total_audio.extend(dolphin_sound)                
                
                file_name = file_name[:file_name.find('.wav')]               


                dolphin_image = convert_to_spectrograms(dolphin_sound, dpi)
                temp_total_image.extend(dolphin_image)
       
    
                pickle.dump(dolphin_image, open(os.path.join(path,folder)+'/'+file_name+'_img.pkl', "wb" ))


               
                line = fp.readline() 



def extract(species_dir,soundscape_dir,species_dir_val,soundscape_dir_val,
                            sample_rate, time_to_extract, augmentation_probability, 
                            augmentation_amount, seed, number_iterations,decision_val,dpi,roots_CetusID):
  
     print('~ CetusID ~')
     print()
     print()
     
     print('Preparing training files: ...')        
     execute_listing_folders(species_dir)
     execute_listing_folders(soundscape_dir)
     print('Training files written.')
     print()
     
     
     dolphin_audio = []
     numbers = []#find the maximim number of samples per class
     for path, folders, files in os.walk(species_dir):
       
       for folder in folders:
        print()
        print('----------------------------------')
        print('Extracting audio data of %s' % os.path.basename(folder))

        species = []
         

        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))

                
                file_name_x = '/'+file_name
                
                extracted_d = execute_audio_extraction_dolphin(os.path.join(path,folder),file_name_x, sample_rate, time_to_extract)
                dolphin_audio.extend(extracted_d)
                species.extend(extracted_d)
                
                
                line = fp.readline() 

        
        number = int(len(species))
        numbers.append(number)
        

                
     max_class_samples = max(numbers)
     print('Maximim amount of data for a class: ', max_class_samples)   

     dolphin_audio = np.asarray(dolphin_audio)
     with open(roots_CetusID + '/Augmentation_load.py', "w") as f:
      f.write('max_class_samples = {}' .format(max_class_samples) +'\n')
     if decision_val == 0:
       extract_validation(species_dir_val,soundscape_dir_val,
                            sample_rate, time_to_extract, augmentation_probability, 
                            augmentation_amount, seed, number_iterations, dpi)
	
def augment(species_dir,soundscape_dir,
                            sample_rate, time_to_extract, augmentation_probability, 
                            augmentation_amount, seed, number_iterations,decision_val,dpi,max_class_samples):

     
     print('Loading soundscape data:')

     soundscape = []  

     for path, folders, files in os.walk(soundscape_dir):
       for folder in folders:
        print()
        print('----------------------------------')
        print('Extracting audio data from %s' % os.path.basename(folder))
                
        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))
                
                file_name_x = '/'+file_name
                
                extracted_s = execute_audio_extraction_soundscape(os.path.join(path,folder),file_name_x, sample_rate, time_to_extract)
                
                soundscape.extend(extracted_s)

               
                line = fp.readline() 

     soundscape = np.asarray(soundscape)
     print()
     print('----------------------------------')
     print ('Soundscape shape:', soundscape.shape)
     print('----------------------------------')

     total_dolphin_augmented = []

     for path, folders, files in os.walk(species_dir):
       
       for folder in folders:
        print()
        print('----------------------------------')
        print('Augmenting data of %s' % os.path.basename(folder))
        
       
        temp_total_augmented = []
        temp_total_image = []
       
        species = []
        numbers_files = []
         

        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))


                file_name = file_name[:file_name.find('.wav')]
                
                extracted_d = pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_audio.pkl', "rb" ))
                number = len(np.asarray(extracted_d))
                numbers_files.append(number)
                
                species.extend(extracted_d)
                
                
                line = fp.readline() 

        species = np.asarray(species)
        species_num = int(species.shape[0])
        print(species_num)
        augmentation_amount_spec = int(augmentation_amount * max_class_samples // species_num)
        max_file_num = max(numbers_files)
        print(max_file_num)
        
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))

                file_name = file_name[:file_name.find('.wav')]
                
                extracted_d = pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_audio.pkl', "rb" ))
                
                file_num = int(len(extracted_d))


                augmentation_amount_spec_files = int(augmentation_amount_spec * max_file_num // file_num)
                





                dolphin_augmented = execute_augmentation(extracted_d, soundscape, time_to_extract, 
                                                         sample_rate, augmentation_amount_spec_files, augmentation_probability, seed, file_name, os.path.join(path,folder))
                temp_total_augmented.extend(dolphin_augmented)
                dolphin_augmented_image = convert_to_spectrograms(dolphin_augmented,dpi)
                temp_total_image.extend(dolphin_augmented_image)
                pickle.dump(dolphin_augmented_image, open(os.path.join(path,folder)+'/'+file_name+'_augmented_img.pkl', "wb" ))
                os.remove(os.path.join(path,folder)+'/'+file_name+'_audio.pkl')
                
                
                line = fp.readline()         
       
    
        
        
        
        temp_total_augmented = np.asarray(temp_total_augmented)
        total_dolphin_augmented.extend(temp_total_image)
        temp_total_image = np.asarray(temp_total_image)       
        print('..~.....~.....~...|||||||||||..~.....~.....~...')
        print()
        print('  ~ %s:' % os.path.basename(folder))
        print()
        print('     Total audio shape:', species.shape)
        print('     Total augmented audio shape:', temp_total_augmented.shape)
        print('     Total image shape:', temp_total_image.shape)
        print('...............................................')

        

        del species, dolphin_augmented, dolphin_augmented_image, temp_total_augmented, temp_total_image 

     

     for path, folders, files in os.walk(soundscape_dir):
       for folder in folders:
        print()
        print('----------------------------------')
        print('Augmenting audio data from %s' % os.path.basename(folder))
                
        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))
                
                file_name = file_name[:file_name.find('.wav')]
                extracted_s = pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_audio.pkl', "rb" ))
                


                soundscape_augmented = execute_augmentation(extracted_s,soundscape, time_to_extract, sample_rate, 
                                                              augmentation_amount_soundscape, augmentation_probability, seed, file_name, os.path.join(path,folder))
                  
                


                soundscape_augmented_image = convert_to_spectrograms(soundscape_augmented,dpi)
       
    
                pickle.dump(soundscape_augmented_image, open(os.path.join(path,folder)+'/'+file_name+'_augmented_img.pkl', "wb" ))
                os.remove(os.path.join(path,folder)+'/'+file_name+'_audio.pkl')

                
                line = fp.readline()

        del extracted_s, soundscape_augmented, soundscape_augmented_image
        
     print('Done.')
     

def augment_soundscape(species_dir,soundscape_dir,
                            sample_rate, time_to_extract, augmentation_probability, 
                            augmentation_amount, seed, number_iterations,decision_val,dpi,max_class_samples):

     
     print('Loading soundscape data:')

     soundscape = []  

     for path, folders, files in os.walk(soundscape_dir):
       for folder in folders:
        print()
        print('----------------------------------')
        print('Extracting audio data from %s' % os.path.basename(folder))
                
        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))
                
                file_name_x = '/'+file_name
                
                extracted_s = execute_audio_extraction_soundscape(os.path.join(path,folder),file_name_x, sample_rate, time_to_extract)
                
                soundscape.extend(extracted_s)

               
                line = fp.readline() 

     soundscape = np.asarray(soundscape)
     print()
     print('----------------------------------')
     print ('Soundscape shape:', soundscape.shape)
     print('----------------------------------')

     

     for path, folders, files in os.walk(soundscape_dir):
       for folder in folders:
        print()
        print('----------------------------------')
        print('Augmenting audio data from %s' % os.path.basename(folder))
                
        
        training_file = os.path.join(path,folder)+'/Training_Files.txt'
         
        with open(training_file) as fp:
            line = fp.readline()

            while line:

                file_name = line.strip()
                print()
                
                print ('Reading file: {}'.format(file_name))
                
                file_name = file_name[:file_name.find('.wav')]
                extracted_s = pickle.load(open(os.path.join(path,folder)+'/'+file_name+'_audio.pkl', "rb" ))
                


                soundscape_augmented = execute_augmentation(extracted_s,soundscape, time_to_extract, sample_rate, 
                                                              augmentation_amount_soundscape, augmentation_probability, seed, file_name, os.path.join(path,folder))
                  
                


                soundscape_augmented_image = convert_to_spectrograms(soundscape_augmented,dpi)
       
    
                pickle.dump(soundscape_augmented_image, open(os.path.join(path,folder)+'/'+file_name+'_augmented_img.pkl', "wb" ))
                os.remove(os.path.join(path,folder)+'/'+file_name+'_audio.pkl')

                
                line = fp.readline()

        del extracted_s, soundscape_augmented, soundscape_augmented_image
        
     print('Done.')
	
     