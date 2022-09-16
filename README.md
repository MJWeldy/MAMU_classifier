This project implements a generalized pilot classification training and inference procedure for audio classification. 

This version of the repository uses BirdNET (Kahl et al. 2021) to embed audio files simulated using the scaper package (Salamon et al. 2017) and trains a new MLP classifier head on top of the BirdNet embeddings.

This project uses Python v 3.10.0. 

To use this project first install pyenv and poetry. 

Instructions for use:
1. Use shell commands navigate to the repository location
2. use pyenv to install python v 3.10.0
    pyenv install 10.0.0
3. Make python 3.10.0 locally available with pyenv 
    pyenv local 10.0.0
4. Use poetry to install dependencies
    poetry install

To run inference on a folder of wav files use poetry to run 'file_inference.py'
    poetry run python main.py v_0 P:\PROJECTS\HJA_BirdNET_transfer\data\test_audio\
        The arguments are the model version number and the path to the audio data

To train a new classifier
    create folders at 'data/foreground' and 'data/background' and populate those folders with audio following the instructions found in the scaper tutorial
    https://scaper.readthedocs.io/en/latest/tutorial.html

Then use poetry to run 'train_classifier.py'
    poetry run train_classifier.py 2500 128 v_0
    The arguments define the number of simulation steps, the number of samples per simulation step, and the version of the model to save. 
    


