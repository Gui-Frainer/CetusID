import sys
roots_CetusID = input("Enter the path to CetusID folder (example:   ): ")
sys.path.append(roots_CetusID)
from Initialization import *
from Load_Train_CNN1 import *
from Load_Train_CNN2 import *
from CNN_networks import *

train_model_CNN2(1, species_dir,species_dir_val,network_CNN2, plot_confusion_matrix, decision_val,location_models,prediction_folder)
print('Done')


