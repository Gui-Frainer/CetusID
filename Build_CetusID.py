import keyboard
import os
import sys

def initialize(roots_CetusID):
  roots_CetusID = roots_CetusID
  if os.path.exists(roots_CetusID):
      if not os.path.exists(roots_CetusID + '/Data'):    
        os.makedirs(roots_CetusID + '/Data')  
      if not os.path.exists(roots_CetusID + '/Data/Species'):
        os.makedirs(roots_CetusID + '/Data/Species') 
      if not os.path.exists(roots_CetusID + '/Data/Soundscape'):
        os.makedirs(roots_CetusID + '/Data/Soundscape')
      if not os.path.exists(roots_CetusID + '/PAM_files'):
        os.makedirs(roots_CetusID + '/PAM_files')
      if not os.path.exists(roots_CetusID + '/Models'):
        os.makedirs(roots_CetusID + '/Models')
    
  decision_val = int(input("Do you want to validate your model (0) with distinct files or (1) splitting your training dataset?:       [0/1]"))

  if decision_val > 1:
    decision_val = int(input("Wrong. 0 or 1?"))
  if decision_val == 0:
    if not os.path.exists(roots_CetusID + '/Data/Validation_dataset'):
      os.makedirs(roots_CetusID + '/Data/Validation_dataset')
    if not os.path.exists(roots_CetusID + '/Data/Validation_dataset/Species'):
      os.makedirs(roots_CetusID + '/Data/Validation_dataset/Species')
    if not os.path.exists(roots_CetusID + '/Data/Validation_dataset/Soundscape'):
      os.makedirs(roots_CetusID + '/Data/Validation_dataset/Soundscape')
    

  if decision_val == 0:
    species_dir_val = roots_CetusID + '/Data/Validation_dataset/Species/'
    soundscape_dir_val = roots_CetusID + '/Data/Validation_dataset/Soundscape/'   

  sample_rate = int(input("Enter the sample rate used for the recordings in kilohertz (kHz) : ")) * 1000 
  time_to_extract = int(input("Enter the time to extract the sounds in seconds (s) (i.e., window size): ")) 
  dpi= int(input("Enter the resolution of the images in dpi: "))

  

  
  species_dir = roots_CetusID + '/Data/Species/' 
  soundscape_dir = roots_CetusID + '/Data/Soundscape/'
  testing_folder = roots_CetusID + '/PAM_files/'
  prediction_folder = roots_CetusID + '/'
  location_models = roots_CetusID + '/Models/'

  
  print()
  if decision_val == 0:
    
    with open(roots_CetusID + '/Initialization.py', "w") as f:
      f.write('sample_rate = {}' .format(sample_rate) +'\n')
      f.write('time_to_extract = {}' .format(time_to_extract) +'\n')
      f.write('dpi = {}' .format(dpi) +'\n')
      f.write('species_dir = "{}"' .format(species_dir) +'\n')
      f.write('soundscape_dir = "{}"' .format(soundscape_dir) +'\n')
      f.write('species_dir_val = "{}"' .format(species_dir_val) +'\n')
      f.write('soundscape_dir_val = "{}"' .format(soundscape_dir_val) +'\n')
      f.write('decision_val = {}' .format(decision_val) +'\n')
      f.write('testing_folder = "{}"' .format(testing_folder) +'\n')
      f.write('prediction_folder = "{}"' .format(prediction_folder) +'\n')
      f.write('location_models = "{}"' .format(location_models) +'\n')
      f.write('roots_CetusID = "{}"' .format(roots_CetusID) +'\n')
    print('CetusID set. Move your training files and validation files accordingly.')
  if decision_val == 1:

    with open(roots_CetusID + '/Initialization.py', "w") as f:
      f.write('sample_rate = {}' .format(sample_rate) +'\n')
      f.write('time_to_extract = {}' .format(time_to_extract) +'\n')
      f.write('dpi = {}' .format(dpi) +'\n')
      f.write('species_dir = "{}"' .format(species_dir) +'\n')
      f.write('soundscape_dir = "{}"' .format(soundscape_dir) +'\n')
      f.write('species_dir_val = None' +'\n')
      f.write('soundscape_dir_val = None' +'\n')
      f.write('decision_val = {}' .format(decision_val) +'\n')
      f.write('testing_folder = "{}"' .format(testing_folder) +'\n')
      f.write('prediction_folder = "{}"' .format(prediction_folder) +'\n')
      f.write('location_models = "{}"' .format(location_models) +'\n')
      f.write('roots_CetusID = "{}"' .format(roots_CetusID) +'\n')

    print('CetusID set. Move your training files accordingly.')