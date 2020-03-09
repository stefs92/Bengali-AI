# Bengali Grapheme Classification: Testing Different Approaches
### Midway Blog Post  -  Mar. 09, 2020

# Introduction
For this blog post, we have implemented some changes in preprocessing the data and tried several incremental changes to our baseline neural network architecture. Instead of jumping right into some of the high-grade CNN architectures available online, we wanted to build up a decently performing model from scratch, and then use one or two fancier tricks to improve the accuracy.

In order to preprocess the data more efficiently, we have started using the ImageDataGenerator class, which allowed us to load the dataset "in real time" while training, hence avoiding overloading the working memory. The dataset is initially available in four .parquet files, each containing around 5000 training images. We loaded the .parquet files, one piece at the time, and used them to generate and save separate image files. The ImageDataGenerator class comes with two tools for loading the data - "FlowFromDirectory" and "FlowFromDataframe" - the first one requiring images of different classes to be stored in separate folders and the second one taking in a pandas dataframe containing the file names and corresponding labels. We opted for the second one, implementing which turned out to be significantly simpler. We took advantage of the methods for rescaling the data and splitting it into the training (80%) and validation (20%).

Since finishing our last blog post, we have realized that increasing the number of filters in convolutional layers can significantly improve the performance. However, when more than 25 filters were used, this resulted in significant overfitting. This is illustrated below, we can see the validation accuracy plateauing as the training accuracy is steadily improving.

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76132697-35ea0c00-5fe2-11ea-881e-02bda7e403ba.PNG">
</p>

# Approach 1: Dropouts

We have attempted to regularize the model by introducing dropouts after max pooling layers. After adding a dropout of 0.5 after each max pooling layer, the model performed significantly worse, with accuracy hovering around 2.5% after 5 epochs - this value was too high. We tuned both the number of filters and the dropout parameter by training the model for several epochs and choosing the best - performing model, with dropouts of 0.1 after max pooling layers and a dropout of 0.2 between the two dense layers at the very end of the neural network. This gave us the validation accuracy of around 47% and the training is visualized in the Tensorboard's graph below (similar to value accuracy)

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76132939-82821700-5fe3-11ea-90cb-9e39500aff20.PNG">
</p>


# Approach 2: Spatial Dropouts with L1 and L2 Reglarizers

For our second approach, we tried to regularize the neural net using SpatialDropouts, a technique that drops 2D Feature maps. <sup>[1]</sup> The following block of code (from TensorFlow) is the standard of how the technique is implemented <sup>[1]</sup>:

```
tf.keras.layers.SpatialDropout2D(
    rate, data_format=None, **kwargs
)
```

The "kwargs" command was giving us a hard time and resulting in errors in our code, so we decided to remove the command, and added the spatial dropout command whenever a Convoultion 2D layer was used:

```
keras.layers.Conv2D(filters=25, kernel_size=2, activation='relu', padding="SAME",
                kernel_regularizer=regularizers.l1(0.01)),
tf.keras.layers.SpatialDropout2D(rate = 0.2, data_format=None),
```

Our final code had 23 layers (5 Convolution 2D, 3 Dense, 4 Dropout, 1 Flatten, 4 MaxPooling, 6 SpatialDropout2D). The Convultion Layers had filters set to 25, kernel sizes of 3, "relu" activation functions, "SAME" paddings, and regularizers set to 0.01 (3 L1 reglarizers and 2 L2 regularizers). The SpatialDropout2D layers had rates of 0.2, the MaxPooling2D layers had pool_sizes set to 2, and the Desnse layers had units of 168, kernel_initializers set to "glorot_normal", 2 activations set to "relu" and 1 set to "softmax" and regularizes set to 0.01 (1 L1 regularizer and 2 L2 regularizers).

After running our model for 30 epochs, we got small values for accuracy and validation accuracy, fluctating between 2.5% and 3%:

<img width="499" alt="spatial dropout approach" src="https://user-images.githubusercontent.com/54907300/76158467-72a22a00-60ec-11ea-9028-eaf247832c72.png">

The validation accuracy (and accuracy) drastically decreased from the 41% validation accuracy our initial model from our initial blog post had, shown below:

<img width="352" alt="initial model" src="https://user-images.githubusercontent.com/54907300/76158482-936a7f80-60ec-11ea-97bf-363855f4539d.png"> 

Our initial model had 15 layers (5 Convolution 2D, 3 Dense, 2 Dropout, 1 Flatten, 4 MaxPooling) and ran for 50 epochs, but adding 8 layers (6 SpatialDropout2D, 2 Dropout) caused the validation accuracy to plummet. Even though the model will perform poorly with this accuracy, at least we know it's not overfitted, since the accuracy and validation accuracy are within the same range (according to a user from StackOverFlow <sup>[2]</sup>).

# Appraoch 3: Exclude Spatial Drop Outs, Keep Regularizers Consistent (L1)

To see if the spatial drop outs and varied regularizers caused the huge decrease in accuracy, we decided to make the neural network a little less dense by excluding the SpatialDropout2D layers and keeping the regularizers consistent (L1). We used 17 layers - 5 Convolution 2D, 3 Dense, 4 Dropouts, 1 Flatten, 4 MaxPooling (we also added 2 dropout layers), keeping all other parameters the same (i.e. filters set to 25). After running the model for 30 epochs, we surprisingly get the same accuracy and validation accuracy, less than 3%. 

<img width="499" alt="spatial dropout approach" src="https://user-images.githubusercontent.com/54907300/76159806-69b85500-60fa-11ea-9386-8836fac8e34e.png">


# Next Steps
 
Our three approaches show that adding more layers to a neural network does not necesarily improve it, but can rather signifcantly decrease the accuracy. It would be interesting to find a threshold of what number of layers is too dense for the model, for it to perform poorly. 

We're also looking into the possibility of implementing embedded arithmetics (based on lecture on 2/25), to see if the Bengali graphemes can be mapped to a vector, and then broken into three components (roots, vowels, consonants) based on similar vector sequences. 

In the remainder of this project, we will try to experiment with more interesting architectures such as dilated convolutions, and perhaps resnets.


# References
1. tf.keras.layers.SpatialDropout2D | TensorFlow Core v.2
2. StackOverFlow | https://stackoverflow.com/questions/51335133/keras-how-come-accuracy-is-higher-than-val-acc
                

