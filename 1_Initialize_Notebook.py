import os
import sys
roots_CetusID = input("Enter the path to CetusID folder (example:   ): ")
sys.path.append(roots_CetusID)
from Build_CetusID import *
initialize(roots_CetusID)
print('Done')



