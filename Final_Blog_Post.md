# Bengali Grapheme Classification: Final Model
### Final Blog Post  -  Mar. 08, 2020

# Introduction

Prior to training the final model, we found a bounding box for each image, and then cropped and resized all images to the same size of (50,100) pixels using skimage.transform library. Initially, all pixels have integer values ranging from 0 to 255, with the "empty" pixels taking values close to 255. For the purposes of finding the bounding box, we experimented with different thresholds, and found that removing rows and columns containing only values greater then 200 works pretty well. The cropped images were saved to new .parquet files using pandas.

Since using an instance of ImageDataGenerator class was sometimes causing the session to crash, in our final work we decided to revert back to loading the four cropped image datasets piecewise. We loaded and trained on datasets separately for a number of iterations, with an EarlyStopping callback function. When accuracy would start decreasing, we would switch to another image dataset and repeat the process a number of times. We used 5000 images from each of the four sets for validation and the remaining roughly 45000 for training.

We have previously noticed that our model started to perform significantly worse when additional convolutional layers were added. For our final model, we have attempted to get some additional performance by making our neural network 3 layers deeper while introducing a ResNet - style connection short-circuiting the additional layers. Our implementation of the residual block was inspired by [1] and the example from our last homework, in which we learned how to customize Keras models using a bit of Tensorflow backend. Since our convolutions had stride = 1 and 'same' padding, the dimension of their output was the same as the input dimension (except the first convolution, which introduced a number of filters). This fact made combining the two tensors at the end of the residual block particularly simple.

The summary of our final model is

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76186045-81014c00-61a7-11ea-8a84-ec434d9ccf7a.png">
</p>

Where all convolutional layers had 20 filters of dimensions 3x3 and relu activations. Prior to writing the midway blog post, we performed extensive tuning of the number of filters; after exploring several different values with our new model, we found that the same values produced the best performance this time as well.

Since we were loading the data piecewise while training so as not to overload the working memory, the performance graphs are pieced together from different training sessions. In order to do so, we used Tensorboard's "Wall" option. The performance of our final model on predicting grapheme roots is shown below,

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76186411-9cb92200-61a8-11ea-9681-ce45de4d7569.PNG">
</p>

Where the x-axis corresponds to training epochs and two different plots correspond to training and validation accuracies. During the majority of the training process, our training accuracy was actually lagging behind the validation accuracy. We believe that this is due to the fact that our model contains dropouts, which are only used for training and not for testing and validation. Our validation accuracy for grapheme roots ended up hovering around 70%.

After training on the grapheme roots, we replaced the last layer of the model with the one appropriate for predicting vowel and consonant diacritics (with 11 and 7 outputs, respectively). Training the neural network on vowel diacritics resulted in the following performance graph, 

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76188824-806cb380-61af-11ea-9774-82076c202865.PNG">
</p>

where our final validation accuracy ended up being around 90%. In this plot, validation accuracy plots are again higher than the ones for training accuracy.

Training our model on consonant diacritics instead gave us the accuracy of around 93% and resulted in the following piecewise performance graph,

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76189139-51a30d00-61b0-11ea-9fa7-ab741628c8dd.PNG">
</p>



# References

[1] Implementation of the Residual Block at https://github.com/relh/keras-residual-unit/blob/master/residual.py
