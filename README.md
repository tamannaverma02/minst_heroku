# Activat env
source env/Scripts/activate

# export variables 
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=2
# Run the application
python run.py

