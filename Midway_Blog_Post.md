# Bengali Grapheme Classification: Testing Different Approaches
### Midway Blog Post  -  Mar. 05, 2020

# Introduction
For this blog post, we have implemented some changes in preprocessing the data and tried several incremental changes to our baseline neural network architecture. Instead of jumping right into some of the high-grade CNN architectures available online, we wanted to build up a decently performing model from scratch, and then use one or two fancier tricks to improve the accuracy.

In order to preprocess the data more efficiently, we have started using the ImageDataGenerator class, which allowed us to load the dataset "in real time" while training, hence avoiding overloading the working memory. The dataset is initially available in four .parquet files, each containing around 5000 training images. We loaded the .parquet files, one piece at the time, and used them to generate and save separate image files. The ImageDataGenerator class comes with two tools for loading the data - "FlowFromDirectory" and "FlowFromDataframe" - the first one requiring images of different classes to be stored in separate folders and the second one taking in a pandas dataframe containing the file names and corresponding labels. We opted for the second one, implementing which turned out to be significantly simpler. We took advantage of the methods for rescaling the data and splitting it into the training (80%) and validation (20%).

Since finishing our last blog post, we have realized that increasing the number of filters in convolutional layers can significantly improve the performance. We have also experimented with making our neural network deeper. Both changes resulted in significant overfitting. This is illustrated below, we can see the validation accuracy plateauing as the training accuracy is steadily improving.

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76132697-35ea0c00-5fe2-11ea-881e-02bda7e403ba.PNG">
</p>

# Approach 1: 

We have attempted to regularize the model by introducing dropouts after max pooling layers. After adding a dropout of 0.5 after each max pooling layer, the model performed significantly worse, with accuracy hovering around 2.5% after 5 epochs - this value was too high. We tuned both the number of filters and the dropout parameter by training the model for several epochs and choosing the best - performing model. This gave us the validation accuracy of around 47% and the training is visualized in the Tensorboard's graph below

<p align="center">
<img width="400" alt="high_level_picture" src="https://user-images.githubusercontent.com/31740043/76132939-82821700-5fe3-11ea-90cb-9e39500aff20.PNG">
</p>


# Approach 2:

For our second approach, we tried to regularize the neural net using SpatialDropouts, a technique that drops 2D Feature maps. <sup>[1]</sup> The following block of code (from TensorFlow) is the standard of how the technique is implemented <sup>[1]</sup>:

```
tf.keras.layers.SpatialDropout2D(
    rate, data_format=None, **kwargs
)
```

The "**kwargs" command was giving us

# Appraoch 3: 

Here, we tried to regularize using L1 and L2 regularization ...


# Next Steps

In the remainder of this project, we will try to experiment with more interesting architectures such as dilated convolutions, and perhaps resnets ...


# References
1. tf.keras.layers.SpatialDropout2D | TensorFlow Core v.2



