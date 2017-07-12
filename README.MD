
Developed with Python 2.7 and Tensorflow 1.0
--------------------------------------------

Note: This code uses spectrogram files stored in the .tif format. Spectrograms are stored as log magnitude,  normalized to [0, 1] so that -80dB corresponds to 0, and 0dB corresponds to 1. 


To run this model, 

1) first get the original raw data.
It is messy, uneven lengths, ogg format, some mono, some stero. 

In some origigData directory, 
pro mpt> git clone  https://github.com/karoldvl/ESC-50.git

2) Next we'll massage the data to make it uniform, and convert the audio files to 
spectrogram files in tiff format. 

Navigate to the utils directory, and open the Jupyter (python) notebook,
ESC50_Convert.ipynb

Set the parameters in the first cell, making sure that the directories you want to use exist. 
Run each cell in sequence to create wave files from the ogg files, and then tiff spectrogram files from wav files. 
This notebook also includes code for generating a label file (text file with list of class lables)


2.5) If you want to add a secondary class to the files to be used for Multi Task Learning (MTL - a kind of regularization), use the Centroid2ndaryClassMaker python notebook (in utils).
	Secondary class ids are tacked on to the file names (foo.tif -> foo._2ndaryID_.tif), and the rest of the pipeline knows how to deal with that. 
    

3) Divide your data into two folders, train and validate 
	a) Go to the data directory for the project.
	   Create two new subdirectories there, train and validate
	   Your home directory for this project should now have a directory stucture:
			./data/esc50spect/[subdirectories of sound classes]
			./data/train
			./data/validate

	b) using either cp --parents (or rsync -R on OSX), copy the data you want from esc50spect to either the train folder or the validate folder. 
		Each sound class is divided into five folds, with file names begining with [1-5]
		So for example, to put 20% of the files in train, and 20% in validate, I do this on my Mac:
			go to ./data/esc50spect/
			prompt> rsync -R */[1-4]*.tif ../train
			prompt> rsync -R */5*.tif ../validate

4) The final step is to turn these data into TFRecord files to be read by the tensorflow training code.
	a) move the label file from the ./data/esc50spect directory to the ./data directory
	b) change directory so that you are in your ./data directory
	c) execute spect2TFRecords.sh  (or spect2TFRecords.2label.sh to include secondary labels for MTL)
		These commands are in the the /scripts subdirectory of the project - I suggest adding the path to your $PATH environment variable to run them from anywhere)
	

5) Now you are ready to back up to the main project directory and train using the TFRecord files for data.

+++++++++++++++

Note: You can test and explore this model with a tiny subset of the ESC-50 sounds provided with the git repository. 
./data2 contains dog and vird sounds, already converted from sounds -> to spectorgrams -> separated into train/ and test/ folders. All you need to do is 
a) navigate to the data2 directory and run the runcmd to convert the data to TFRecords (or runcmd.2label.txt to include secondary labels for MTL), and then 
b) go to the main project folder and run 
   prompt> python DCNSoundClass.py