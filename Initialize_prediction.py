from datetime import datetime, timedelta
from Extract_Augment import build_training_files_list

def get_start_time_from_first_file_soundtrap(prediction_folder):
  training_file = prediction_folder +'Training_Files.txt'
  with open(training_file) as fp:
            line = fp.readline()
            while line:
                file_name = line.strip()
                print()                
                
                name_split = file_name.split(".")
                date = name_split[1]
                
                start_time_deployment = datetime.strptime(date, "%y%m%d%H%M%S")
                print('Deployment starting at {}' .format(start_time_deployment))
                break

  return start_time_deployment


def get_start_time_from_first_file_hydromoth(prediction_folder):
  training_file = prediction_folder +'Training_Files.txt'
  with open(training_file) as fp:
            line = fp.readline()
            while line:
                file_name = line.strip()
                print()                
                
                name_split = file_name.split(".")
                date = name_split[0]
                
                start_time_deployment = datetime.strptime(date, "%Y%m%d_%H%M%S")
                print('Deployment starting at {}' .format(start_time_deployment))
                break

  return start_time_deployment

def set_prediction(roots_CetusID,testing_folder):

  build_training_files_list(testing_folder)
  recorder_type = int(input("Please, enter your recorder device type (0 - Hydromoth/Audiomoth; 1 - Soundtrap):       [0/1]"))
  if recorder_type == 0:
      print('Hydromoth/Audiomoth')
      real_start_time = get_start_time_from_first_file_hydromoth(testing_folder)
  if recorder_type == 1:
      print('Soundtrap')
      real_start_time = get_start_time_from_first_file_soundtrap(testing_folder)
  n_CNN1 = int(input("Enter the model number for the CNN1: "))# example: 811965
  weights_name_CNN1 = 'weights_{}_CNN1.hdf5' .format(n_CNN1) #200x200
  n_CNN2 = int(input("Enter the model number for the CNN2: "))# example: 172484
  weights_name_CNN2 = 'weights_{}_CNN2.hdf5' .format(n_CNN2) #200x200
  batch_number = int(input("Enter the length of the audio file in minutes: "))# example: 811965
  time_within_detections = int(input("Enter the length of the acoustic encounter in minutes: "))*60  
  with open(roots_CetusID + '/Prediction_load.py', "w") as f:
      f.write('weights_name_CNN1 = "{}"' .format(weights_name_CNN1) +'\n')
      f.write('weights_name_CNN2 = "{}"' .format(weights_name_CNN2) +'\n')
      f.write('batch_number = {}' .format(batch_number) +'\n')
      f.write('real_start_time = "{}"' .format(real_start_time) +'\n')
      f.write('time_within_detections = {}' .format(time_within_detections) +'\n')