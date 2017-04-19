export ENV_NAME=kgsealions

# Start the virtualenvironment
if [ ! -f environment.yml ]; then
    conda create -n $ENV_NAME;
else
    conda env create -n $ENV_NAME -f environment.yml 2> /dev/null
fi
source activate $ENV_NAME

## Be sure jupyter is available
conda install jupyter
ipython kernel install --name "Virtulenv_Python_3_$ENV_NAME" --user

# System path variables
export PROJ_PATH=/home/jose/tech/ml_projects/kaggle/sealions_contest
export DATA_PATH=/home/jose/tech/ml_projects/datasets/Kaggle-NOAA-SeaLions_FILES
export DATAMODEL_PATH=/home/jose/tech/ml_projects/data_models/sealions_contest

# Copy exclude files
cp git_exclude .git/info/exclude

# Export virtualenvironment
conda env export > environment.yml


