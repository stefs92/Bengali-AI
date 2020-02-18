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

Although it's a steep task, our team is prepared and has prior experience with image classification, working with the renowned MNIST Dataset (shown below) to organize numbers by different fonts:

<img width="575" alt="Screen Shot 2020-02-18 at 1 47 18 AM" src="https://user-images.githubusercontent.com/54907300/74720496-e941bb00-5203-11ea-9626-bfdd9d10ecb4.png">

# Loading the Data

There are approximately 10,000 grapheme images to work with. 

We will mostly be using the .parquet train and test files, each of which contains tens of thousands of images (size 137 x 236). Each row represents an image, and we plotted one row as a test:

<img width="329" alt="map_94" src="https://user-images.githubusercontent.com/54907300/74727346-04fe8e80-520f-11ea-9693-86e82d1ed432.png">

This image looks similar to the 94th grapheme root, in the table below:

<img width="436" alt="Screen Shot 2020-02-18 at 5 26 45 AM" src="https://user-images.githubusercontent.com/54907300/74727573-6292db00-520f-11ea-8242-8b36604e1408.png">



# Neural Network Model 
We attack this problem by designing a deep convolutional neural network of the following form ...

# Appendicies

### Code

```markdown
Syntax highlighted code block 

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text



[Link](url) and ![Image](src)
```

<!-- For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/). >

<!-- ### Jekyll Themes >

<!-- Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/stefs92/Bengali-AI/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file. >

<!-- ### Support or Contact >

<!-- Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out. >
