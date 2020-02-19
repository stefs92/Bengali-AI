<!-- ## Welcome to GitHub Pages -->

<!-- You can use the [editor on GitHub](https://github.com/stefs92/Bengali-AI/edit/master/index.md) to maintain and preview the content for your website in Markdown files. -->

<!-- Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files. -->

<!-- Hey guys, I commented out some lines with this command, I guess this is GitHub's way of commenting -->

# Bengali Grapheme Classification: Assessing the Challenge
### Blog Post 1  -  Feb. 18, 2020

# Introduction
While being spoken by more than 200 milion people, Bengali language is particulary interesting from the point of view of AI handwritten recognition. Each bengali letter consists of 3 parts -one of 168 possible grapheme roots, one of 11 possible vowel diacritics and one of 7 possible consonant diacritics. The sheer number of combinations makes handwritten symbol recognition a challenging machine learning problem.


At a high level, we wish to break down an image of a Bengali word and assign the pieces to three bins:
<img width="647" alt="high_level_picture" src="https://user-images.githubusercontent.com/54907300/74720359-abdd2d80-5203-11ea-90a5-734785bae48b.png">
 
Although it's a steep task, our team is prepared and has prior experience with image classification that could be helpful, such as working with the renowned MNIST Dataset (shown below) to organize numbers by different fonts:

<img width="575" alt="Screen Shot 2020-02-18 at 1 47 18 AM" src="https://user-images.githubusercontent.com/54907300/74720496-e941bb00-5203-11ea-9626-bfdd9d10ecb4.png">

# Examining the Data

When loading the data, we see there are approximately 10,000 grapheme images to work with. 

We will mostly be using the .parquet train and test files, each of which contains tens of thousands of images (each size 137 x 236). Each row represents an image, and we plotted one row as a trial run:

<img width="436" alt="Screen Shot 2020-02-18 at 5 26 45 AM" src="https://user-images.githubusercontent.com/54907300/74727573-6292db00-520f-11ea-8242-8b36604e1408.png">


We noticed the image has some similarities to the 94th grapheme root from the glossary:

<img width="329" alt="map_94" src="https://user-images.githubusercontent.com/54907300/74727346-04fe8e80-520f-11ea-9693-86e82d1ed432.png">

We believe the more we manually look at the images, the more we can improve our understanding of the Bengali language, which can ultimately help us form our model. 

# Neural Network Model 
We attack this problem by designing a deep convolutional neural network of the following form:

```markdown
heigth = 137;
width = 236;
X_train_full=train_df0.values.reshape(-1,heigth,width,1);
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train_full = train_df_['grapheme_root'][:50210];
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

model = keras.models.Sequential([
    keras.layers.MaxPooling2D(pool_size=2,input_shape=[heigth, width, 1]),
    keras.layers.Conv2D(filters=8, kernel_size=7, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=3),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.Conv2D(filters=8, kernel_size=3, activation='relu', padding="SAME"),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=168, activation='softmax'),
    
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid))
])

pd.DataFrame(history.history).plot(figsize=(8, 5))
```




<img width="508" alt="history plot" src="https://user-images.githubusercontent.com/54907300/74779447-81bf5600-526b-11ea-9a31-45e220546967.png">

Based on the graph above, the model proves to have high accuracy and low loss values.

# Next Steps

We noticed the images we loaded have a large yellow cloud around the graphemes. To prevent the model from unnecessarily traning this yellow space, we hope to focus the model on just the blue-lined grapheme. This would involve looking at bounding boxes for our images. Cropping to the union of bounding boxes for all images would be a safe bet but might still result in unnecessarily large image size - therefore, we might want to restrict to a box size large enough to cover (100 - p)% of the images where p is a small parameter that can be tuned to increase accuracy.

We can also experiment with different possible ways of training the network. The full dataset seems to large for Keras to handle simultaneously, so the way to train seems to be to split it into 4 pieces and train on each one separately for some number_of_epochs. It is possible that accuracy would increase if we reduce number_of_epochs and then repat the process many times - this way, the neural network would have a chance to look at the entire dataset before getting really good at predicting its subsets.



<!-- For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/). >

<!-- ### Jekyll Themes >

<!-- Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/stefs92/Bengali-AI/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file. >

<!-- ### Support or Contact >

<!-- Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out. >
