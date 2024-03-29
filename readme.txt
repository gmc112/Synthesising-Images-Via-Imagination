Notes on files in submission
- The Highlights folder contains some images handpicked from good runs
- The run folder contains a couple of example of long training runs
- Standard MNIST that I used is in data folder, pokemon images I used are in the used dataset folder

Specifications of System 
To be able to run this software in a reasonable period of time, a GPU is required to be present in the computer used. Due to lack of availability, 
this software has only been tested on an Nvidia GPU using CUDA. Differing system specifications can lead to significantly different training times. 
Running this software without a GPU is not recommended 
The specifications of the system all testing was done on are as follows:
	Intel i7-8700k @ 4.77GHz
	Nvidia GeForce GTX1080 with 2560 CUDA Cores
	32GB DDR4 RAM
	Windows 10 Pro 64-bit
	CUDA version 10.0.132


Installation
1.	Install Python 3.6 available at https://www.python.org/downloads/release/python-360/
2.	Install Anaconda for Python 3 available at https://www.anaconda.com/distribution/. 
3.	Install Pycharm by Jetbrains available at https://www.jetbrains.com/pycharm/download/#section=windows
4.	Run Anaconda and enter the following commands to create the environment that will be used to run the software:
		conda create –name tf-gpu python=3.6
		activate tf-gpu
5.	Now it is time to install the packages used. Enter the commands:
		conda install tensorflow-gpu
		conda install tensorboard
	Follow through with the installation process until complete.
6.	Open the Pycharm and create a new project. Select the python.exe from the created Anaconda environment as the project Interpreter. The default location for this to be created at is “C:\ProgramData\Anaconda3\envs\tf-gpu”
7.	Go to File>Settings>Project>Project Interpreter and click the + symbol in the top right hand corner. Search for and install the following packages:
		numpy
		imageio
		Pillow
8.	Copy the files provided with software to the project directory
9.	Two variables in each of the files Gan.py, Gan2.py, Gan3.py, DAE.py, and Vae.py need to be altered before the networks can be run: SAVE_PATH in the block of global variables at the top and the variable output in the block at the bottom. Both of these need to be set to a valid location present on the computer being used.



Using the System
The models can be run by opening each of the files (Gan.py, Gan2.py, Gan3.py, DAE.py, and Vae.py) and right clicking the main code area in PyCharm and selecting Run. 

The variable RESTORE should be set to False if a brand-new run is to be started, otherwise if it is set to True and an earlier run of that network is detected, 
that run will then be loaded and have its training continued.

The training data can be selected by placing folders of images in the data folder in the project directory. 
The multiple datasets used are present in the used dataset folder. Other datasets of images can be used. 
The only requirement being that the images are either all in PNG or all in JPG formats. 
If the images used are PNGs, the global variable MODE should be set to 0. If the images used are JPGs, the global variable MODE should be set to 1.

Altering hyper-parameters can be done by altering the global variables at the top of each file. The recommended variables to change for testing are as follows:
	LEARNING_RATE
	DATA_REPEATS
	EPOCHS
	DROPOUT_RATE where applicable
	DISTORTED

