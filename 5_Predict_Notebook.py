import sys
roots_CetusID = input("Enter the path to CetusID folder (example: /content/drive/MyDrive/CetusID): ")
sys.path.append(roots_CetusID)
from Initialization import *
from Initialize_prediction import *
from CNN_networks import *
from Extract_Augment import convert_to_spectrograms, build_training_files_list, execute_listing_folders
from Predict import *
set_prediction(roots_CetusID,testing_folder)
from Prediction_load import *
execute_processing(testing_folder,sample_rate, location_models, weights_name_CNN1, weights_name_CNN2,
                   time_to_extract, prediction_folder, convert_to_spectrograms, build_training_files_list,
                   execute_listing_folders,batch_number,network_CNN1, network_CNN2,network_CNN1_TF, network_CNN2_TF,species_dir,dpi,real_start_time,time_within_detections, verbose = True)
print('Done')