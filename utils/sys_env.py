import os

def set_env():
    #os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # suprime avisos de nível INFO e WARNING
