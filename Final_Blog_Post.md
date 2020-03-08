# Bengali Grapheme Classification
### Final Blog Post  -  Mar. 08, 2020

# Introduction

Since ImageDataGenerator class was causing some unncecessary issues, in our final work we decided to revert back to loading the four image datasets piecewise. We loaded and trained on datasets separately for a number of iterations, with an EarlyStopping callback function. When accuracy would start decreasing, we would switch to another image dataset and repeat the process a number of times.

# References

[1] Implementation of the Residual Block at https://github.com/relh/keras-residual-unit/blob/master/residual.py
