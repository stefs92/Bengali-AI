# Bengali Grapheme Classification: Testing Different Approaches
### Midway Blog Post  -  Mar. 05, 2020

# Introduction
For this blog post, we have implemented some changes in preprocessing the data and tried several incremental changes to our baseline neural network architecture. Instead of jumping right into some of the high-grade CNN architectures available online, we wanted to build up a decently performing model from scratch.

In order to preprocess the data more efficiently, we have started using the ImageDataGenerator class, which allowed us to load the dataset "in real time" while training, hence avoiding overloading the working memory. The dataset is initially available in four .parquet files, each containing around 5000 training images. We loaded the .parquet files, one piece at the time, and used them to generate and save separate image files. The ImageDataGenerator class comes with two tools for loading the data - "FlowFromDirectory" and "FlowFromDataframe" - the first one requiring images of different classes to be stored in separate folders and the second one taking in a pandas dataframe containing the file names and corresponding labels. We opted for the second one, implementing which turned out to be significantly simpler.


# Approach 1: 

The...

# Approach 2:

The...

# Appraoch 3: 

The...

# Next Steps
The....

# References

The...
