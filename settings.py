import os
import sys

# Pull paths from the environment. Run config.repo.sh to set these variables
CARS_PATH=os.environ['CARS_PATH']
DATA_PATH=os.environ['DATA_PATH']
DATAMODEL_PATH=os.environ['DATAMODEL_PATH']

# Add the path to the pythonpath
sys.path.append(CARS_PATH)