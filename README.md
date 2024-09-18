# AI Card Project
This project is a small part of a much larger project. It analyzes images of baseball cards and tries to determine where the creases are. The following guide helped me a lot. 
* https://christianjmills.com/posts/pytorch-train-mask-rcnn-tutorial/

# Install dependencies
* run the following command
* pip install -r requirements.txt
* Install CUDA version 12.1 (You'll need an Nvidia GPU)

# How to Evaluate Image
* There's a trained model in /Checkpoints that will be used for evaluating
* run the following command
* python ./evaluate_image.py "path/to/image"

# How to Create Training Data
* The trained model uses the images and annotation in /Images
* I used a program called Labelme to annotate each image

# How to Train Model
* It uses the images and annotations in /Images for training
* python ./train_model.py

# Screenshots
![](README/Example.PNG)

# Built With
* Python
* PyTorch

# Authors
* Richard Vo
