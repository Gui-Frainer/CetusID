import keyboard
import time
import numpy as np
import pandas as pd
import soundfile as sf
from os import listdir
import librosa
import collections
import time
import tarfile
import pandas as pd
import soundfile as sf
from os import listdir
import librosa
import collections
import datetime
import os
import time
import pickle
from datetime import datetime, timedelta


import keyboard
def execute_processing(testing_folder,sample_rate, location_models, weights_name_CNN1, weights_name_CNN2, time_to_extract, prediction_folder, convert_to_spectrograms, build_training_files_list, execute_listing_folders, batch_number, network_CNN1, network_CNN2,network_CNN1_TF, network_CNN2_TF,species_dir,dpi,real_start_time,time_within_detections, verbose):
  
   
    start = time.time()
    start_reading = time.time()

    print('~ CetusID ~')
    print()
    print('Starting predicting PAM files...')    
    print()

    
    
    model_predictions_CNN1 = []
    model_predictions_CNN2 = []
    start_times = []
    end_times = []
    
    
    testing_files = testing_folder +'/Training_Files.txt'
    real_time = 0
    with open(testing_files) as fp:
            line = fp.readline()
            while line:
                file_name = line.strip()
                print()                
                print ('Reading file: {}'.format(file_name))
                #print ('Reading audio file (this can take some time)...')

                test_file_audio, test_file_sample_rate = librosa.load(testing_folder + file_name, 
                                                          sr=sample_rate)
                
                audio_length = librosa.get_duration(test_file_audio, sample_rate)
                
                amount_of_chunks = int(audio_length - time_to_extract+1)
                
                
                model_prediction_CNN1, model_prediction_CNN2 = execute_batches(test_file_audio, audio_length, file_name, time_to_extract, sample_rate, location_models, weights_name_CNN1, weights_name_CNN2, batch_number,testing_folder,convert_to_spectrograms,network_CNN1,network_CNN2,network_CNN1_TF, network_CNN2_TF,species_dir,dpi)
                model_predictions_CNN1.extend(model_prediction_CNN1)

                
                
                model_predictions_CNN2.extend(model_prediction_CNN2)

                

                start_times_files, end_times_files = create_time_index(time_to_extract, batch_number, real_time, real_start_time) 
                
                
                start_times.extend(start_times_files)
                end_times.extend(end_times_files) 
                real_time = real_time + batch_number*60 #+  time_to_extract - 1#added + time_to_extract - 1
                del start_times_files, end_times_files               
                line = fp.readline()#Switch it on while escalating the analysis
    
    model_predictions_CNN1 = np.asarray(model_predictions_CNN1)
    model_predictions_CNN2 = np.asarray(model_predictions_CNN2)

    model_predictions = np.column_stack((model_predictions_CNN1, model_predictions_CNN2))


    start_times = np.array(start_times)
    end_times = np.array(end_times)
    
    
    results = pd.DataFrame(np.column_stack((start_times, end_times, model_predictions_CNN1[:,0],model_predictions_CNN1[:,1])), columns=['Start(seconds)', 'End(seconds)', 'Pr(absence)', 'Pr(presence)'])

    number = 0
    for path, folders, files in os.walk(species_dir):
      for folder in folders:
        
        
        
        
        class_results = pd.DataFrame(model_predictions_CNN2[:,number], columns=['{}'.format(os.path.basename(folder))])
        
        
        results = pd.concat([results, class_results], axis=1)
        
        number = number + 1    

    
    real_start_time = datetime.fromisoformat(real_start_time)  
    UTC_datetime = []
    for index, row in results.iterrows():
        
        
        
        
        
        
        
        
        time_sec = pd.to_timedelta(row['Start(seconds)'], unit='s')
        
        
      
        
        
        
        
        time_detection = time_sec + real_start_time#datetime.strptime(time, "%Y-%m-%d %H:%M:%S.%f")        
        UTC_datetime.append(time_detection)    
    UTC_datetime = pd.DataFrame(UTC_datetime, columns = ['UTC_datetime'])
    results.insert(0, "UTC_datetime", UTC_datetime, True)
    results.to_csv(prediction_folder + 'predictions_raw.csv', index=False)

    post_process(results,time_within_detections,prediction_folder,0.70)
    
    print ()
    print ('Reading done.')
    end_reading = time.time()

    

def create_time_index(time_to_extract, batch_number, real_time, real_start_time):
    
    start = []
    end = [] 

    
    amount_of_chunks = int(batch_number*60 - time_to_extract + 1)#tirei o +1
    
    
    
    for i in range (0, amount_of_chunks):
        

        start.append(i + real_time)#add to start_time!!
        end.append(time_to_extract + (i) + real_time)#add to start_time!!----------------real clock?
    
    return start, end
    

def execute_batches(audio, audio_length, file_name, time_to_extract, sample_rate, location_models, weights_name_CNN1, weights_name_CNN2, batch_number,testing_folder,convert_to_spectrograms,network_CNN1,network_CNN2,network_CNN1_TF, network_CNN2_TF,species_dir,dpi):
 
    model_prediction_CNN1 = []
    model_prediction_CNN2 = []
    start_index = 0
    
    end_index = 60
    print('checking end index', end_index)
    
    for i in range(batch_number):
        
        print('Processing batch: {} out of {}'.format(i, batch_number))

        batch_prediction_CNN1,batch_prediction_CNN2  = predict(audio, start_index, end_index, time_to_extract, sample_rate, location_models, weights_name_CNN1, weights_name_CNN2,testing_folder,convert_to_spectrograms,network_CNN1,network_CNN2,network_CNN1_TF, network_CNN2_TF,species_dir,dpi)
        
        
        
        model_prediction_CNN1.extend(batch_prediction_CNN1)
        
        model_prediction_CNN2.extend(batch_prediction_CNN2)
        
        
        start_index = end_index - (time_to_extract - 1)
        
        
        end_index = end_index + 60
        

        
        
        
        time.sleep(1)
        

    return model_prediction_CNN1, model_prediction_CNN2
def predict(audio_test, start_index, end_index, time_to_extract, sample_rate, location_models, weights_name_CNN1, weights_name_CNN2,testing_folder,convert_to_spectrograms,network_CNN1,network_CNN2,network_CNN1_TF, network_CNN2_TF,species_dir,dpi):
    

    

    create_X_new(audio_test, 
                         time_to_extract, 
                         sample_rate,start_index, end_index,testing_folder, verbose = False)
        
    X = pickle.load(open(testing_folder+'temp_audio.pkl', "rb" )) #pickle.dump(X, open(testing_folder+'_audio.pkl', "wb" ))
      
    
    
    
    X_spec = convert_to_spectrograms(X,dpi)
    X_spec = np.stack([X_spec,X_spec,X_spec], axis=3)
    X_spec = np.squeeze(X_spec)
    os.remove(testing_folder+'temp_audio.pkl')
    #print(X)
    pickle.dump(X_spec, open(testing_folder+'temp_img.pkl', "wb" ))
    image_size = int(X_spec.shape[1])
    
    
    model_CNN1 = network_CNN1_TF(image_size)
    
    model_CNN1.load_weights(location_models + weights_name_CNN1)

    model_CNN2 = network_CNN2_TF(image_size,species_dir)
    model_CNN2.load_weights(location_models + weights_name_CNN2)
    
    
    model_prediction_CNN1 = model_CNN1.predict(pickle.load(open(testing_folder+'temp_img.pkl', "rb" )))
    model_prediction_CNN2 = model_CNN2.predict(pickle.load(open(testing_folder+'temp_img.pkl', "rb" )))
    
    #print(model_prediction_CNN2)
    del X_spec, X, model_CNN1, model_CNN2

    return model_prediction_CNN1, model_prediction_CNN2

def create_X_new(mono_data, time_to_extract, sample_rate,start_index, end_index,testing_folder, verbose):
    
    X_frequences = []
    time_to_extract_minus_one = time_to_extract - 1

    sampleRate = sample_rate
    duration = int(end_index - start_index - time_to_extract_minus_one)
    if verbose:
        
        print ('-----------------------')
        print ('start (seconds)', start_index)
        print ('end (seconds)', end_index)
        print ('duration (seconds)', (duration))
        print()
    counter = 0
    
    end_index = start_index + time_to_extract
    
    for i in range (0, duration):
    
        if verbose:
            print ('Index:', counter)
            print ('Chunk start time (sec):', start_index)
            print ('Chunk end time (sec):',end_index)
            
        
        extracted = mono_data[int(start_index *sampleRate) : int(end_index * sampleRate)]

        X_frequences.append(extracted)
        
        start_index = start_index + 1
        end_index = end_index + 1
        counter = counter + 1
        
    X_frequences = np.array(X_frequences)
    
    
    pickle.dump(X_frequences, open(testing_folder+'temp_audio.pkl', "wb" ))
    if verbose:
        print ()

    

def post_process(results,time_within_detections,prediction_folder,threshold):
    values = results    
    values['Pr(presence)'] = values['Pr(presence)'].apply(lambda x: 1 if x > threshold else 0)    
    values = values.loc[values['Pr(presence)'] == 1]    
    values_species = set_species(values)    
    values.insert(5, "SpeciesID", values_species, True)    
    values = values[['UTC_datetime', 'Start(seconds)','SpeciesID']]    
    encounters = define_encounters(values,time_within_detections)    
    results = identify_species(encounters)
    results.to_csv(prediction_folder + 'results.csv', index=False)
    
def set_species(values):
    species = values.loc[values['Pr(presence)'] == 1]
    species.drop(['UTC_datetime', 'Start(seconds)','End(seconds)','Pr(absence)', 'Pr(presence)'], axis='columns', inplace=True)
    species['ID'] = species.idxmax(axis=1)
    speciesID = []
    for index, row in species.iterrows():
        ID = row['ID']
        speciesID.append(ID)       
    return speciesID

def define_encounters(values,time_within_detections):
    time_difference = values['Start(seconds)'].diff()
    values.insert(3, "Time_difference", time_difference, True)
    values = values.reset_index()
    values.drop(['index'], axis='columns', inplace=True)
    values.loc[0, 'Time_difference'] = 0    
    values['Time_difference_0'] = values['Time_difference'].apply(lambda x: 0 if x < time_within_detections else x)
    components = []
    comp_number = 1
    i = 0    
    for index, row in values.iterrows():
        if values.loc[i, 'Time_difference_0'] == 0.0:
            components.append('Component {}'.format(comp_number))
        if values.loc[i, 'Time_difference_0'] > 0.0:
            comp_number = comp_number + 1
            components.append('Component {}'.format(comp_number))
            #comp_number = comp_number + 1 
        i = i + 1    
    values['Components'] = components    
    values = values.groupby('Components').filter(lambda x: len(x)>1)#change to 2 when in real world
    values.index=range(0,len(values))
    values.loc[0, 'Time_difference_0'] = 0
    encounters = []
    enc_number = 1
    i = 0     
    for index, row in values.iterrows():
        if values.loc[i, 'Time_difference_0'] == 0.0:
            encounters.append('Encounter {}'.format(enc_number))
        if values.loc[i, 'Time_difference_0'] > 0.0:
            enc_number = enc_number + 1
            encounters.append('Encounter {}'.format(enc_number))
            #enc_number = enc_number + 1
        i = i + 1    
    values['Encounters'] = encounters    
    encounters = values[['Encounters','UTC_datetime','SpeciesID','Start(seconds)']]    
    return encounters
    
def identify_species(encounters):
    
    E = 1
    i = 0    
    encounters_list = []
    species_ID = []
    start = []
    end = []    
    print(' ~ Post-Processing ~')
    print()    
    for i in range(len(encounters['Encounters'].value_counts())):
        
        new_df = encounters.loc[encounters["Encounters"] == 'Encounter {}' .format(E)]        
        encounter_number = 'Acoustic Encounter {}' .format(E)
        encounters_list.append(encounter_number)
        species = new_df['SpeciesID'].value_counts().idxmax()
        species_ID.append(species)
        start_time_n = new_df['Start(seconds)'].iloc[0]
        start_time = new_df['UTC_datetime'].iloc[0]        
        start.append(start_time)
        end_time_n = new_df['Start(seconds)'].iloc[-1]
        end_time = new_df['UTC_datetime'].iloc[-1]
        end.append(end_time)
        duration = int(end_time_n - start_time_n)        
        duration = timedelta(seconds=duration)
        prop = new_df['SpeciesID'].value_counts(normalize=True).to_string()       
        
        
        print('Acoustic Encounter {}:' .format(E))
        print()
        print('Number of detections: {}'.format(len(new_df)))
        print('Encounter duration: {}'.format(duration))
        print('Start time:', new_df['UTC_datetime'].iloc[0])
        print('End time:', new_df['UTC_datetime'].iloc[-1])
        print()
        print('Proportion for each species: ')
        print(prop)
        print()
        print('Species ID:', new_df['SpeciesID'].value_counts().idxmax())
        print()
        print('               ~                    ')
        i = i + 1
        E = E + 1
        
    encounters_list = np.array(encounters_list)    
    species_ID = np.array(species_ID)    
    start = np.array(start)
    end = np.array(end)
    results = pd.DataFrame(np.column_stack((encounters_list, start, end, species_ID)), columns = ['Encounters', 'Start time', 'End time','Species ID'])
    return results