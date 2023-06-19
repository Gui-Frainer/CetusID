import sys
roots_CetusID = input("Enter the path to CetusID folder (example:   ): ")
sys.path.append(roots_CetusID)
from Initialization import *
from Load_Train_CNN1_TransferLearning import *

from CNN_networks import *
train_model_CNN1(1, species_dir, soundscape_dir,species_dir_val,soundscape_dir_val,network_CNN1, plot_confusion_matrix,decision_val,location_models,prediction_folder)

print('Done')


