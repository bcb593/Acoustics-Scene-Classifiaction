# Acoustics-Scene-Classifiaction
Acoustics Scenes Classification Using CNN with Spectrogram Images of sound

Classification of sound can be done in different methods such as using spectrogram and various other signal processing methods. This project is based on acoustics scene Classification using CNN in spectrogram images generated from the sound. 

The dataset used in this project is: TUT Urban Acoustic Scenes 2019 Open set, Development. This dataset contains more than 700 samples for each acoustic scenes while there are 10 different acoustic scenes. 

Use of model: Simple modificaton of VGG-16 is used with 14 convolution layers followed by relu activation, 5 pooling layers(Max Pooling) and 3 densely connected layers which has 1 output layer with softmax activation for classification at the end. 

Used GPU for training the model since there are more than 500 images with 256x256 pixels
