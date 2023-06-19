import os
import sys
roots_CetusID = input("Enter the path to CetusID folder (example:   ): ")
sys.path.append(roots_CetusID)
from Extract_Augment import *
from Initialization import *
extract(species_dir,soundscape_dir,species_dir_val,soundscape_dir_val,
                            sample_rate, time_to_extract, augmentation_probability, 
                            augmentation_amount, seed, number_iterations,decision_val,dpi,roots_CetusID)
print('Done')



