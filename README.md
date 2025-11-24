# ME4990_Final_Project
# Machine Learning Project - Recognizing JPG Numbers

#------------------------PROCCESS------------------------------------------------------------------------------------

# Train a model to recognize hand written numbers. 
# How?
#   Data sets:
#        Use training set for training, use example testing set for evaluation
#        Augment data for a larger training set → more accurate training

#   Use at least 3 networks (one can be from an online source, but cannot be loaded directly - no copy/paste)
#        Show Network structure(layers/functions), and understanding of dimension in each layer
#            EX: Conv(5×5) → ReLU → MaxPool → Conv(5×5) → ReLU → FC → Softmax

#   Find models with the highest accuracy
#        Trained versions of the network (explain weights, bias, learning rate , epochs, optimizers, and loss progress, MSE)
#        Should be saved under a specific file name to use trained model during demo

#   GUI:
#        Display uploaded JPG, and present prediction
#        Have a method to select which one of the 3 trained methods (models) to use for prediction
#        Other information like histogram of probability, accuracy, and confusion matrix from test set
#        Make It Pretty!!

#-----------------------DELIVERABLES------------------------------------------------------------------------------------

#  Progress Report:
#        Use presentation slide (so you can re-use for presentation and report)
#           Show your GUI is working
#           Show your proposed networks you are going to implement
#           Any other items available

#  Demonstration:
#        The instructor will provide you 10 images with the same size in jpg, recognize these 10 images one by one

#  Report: 
#        Intro(Research) → Method/Approach(Data, Network structure/dimensions) → Implementation (Structure of code,          #                  Criterion & optimization methods, # of epochs/batch size, progress of loss with epochs) →                 #                              Result (Testing accuracy & confusion matrix, Analysis) → Conclusion

#--------------------------TIPS------------------------------------------------------------------------------------

# Network
#      Start with simple ones
#      Modify from existing ones
# Load data
#      Takes some time
# During debug stage, don’t load all data
# Make sure your code grammatically works before loading all data
# Image shall be loaded as grayscale
# Pixel intensity is best in the scale of 0.0- 1.0, not 0-255.
# Use ChatGPT to help your loading and encoding
# Train-test Split: 80% train, 20% test (??)
# Training
#    There is no tip here, it takes some time 
#    Print out loss for every 10 or 100 epoch so you monitored your code
#    Plan plenty of time to tune your hyper-parameter
#Evaluation
#    Report accuracy and confusion matrix
#    Through confusion matrix, you may see maybe there is a trend. (E.g. many 6 is miss classified to 0). Then, you can do   #                                                                                            something (what you can do?)
#    Export your prediction and the ground truth to .csv (see our course example) for better understanding your result 
# For demonstration, you don’t have time to train data on-site
# You need to load your pre-trained model!
