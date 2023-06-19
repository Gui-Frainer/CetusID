import os
import sys
roots_CetusID = input("Enter the path to CetusID folder (example:   ): ")
sys.path.append(roots_CetusID)
from Extract_Augment import *
from Initialization import *
from Augmentation_load import *
augment_soundscape(species_dir,soundscape_dir,
                            sample_rate, time_to_extract, augmentation_probability, 
                            augmentation_amount, seed, number_iterations,decision_val,dpi,max_class_samples)
print('Done')



