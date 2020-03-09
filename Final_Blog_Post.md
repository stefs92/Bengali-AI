# Bengali Grapheme Classification: Final Model
### Final Blog Post  -  Mar. 08, 2020

# Introduction

Since using an instance of ImageDataGenerator class was sometimes causing the session to crash, in our final work we decided to revert back to loading the four image datasets piecewise. We loaded and trained on datasets separately for a number of iterations, with an EarlyStopping callback function. When accuracy would start decreasing, we would switch to another image dataset and repeat the process a number of times.

We added a bounding box ...

We added an implementation of residual block inspired by [1] and the example from one of our homeworks, and made the network 3 layers deeper. This resulted in ...

# References

[1] Implementation of the Residual Block at https://github.com/relh/keras-residual-unit/blob/master/residual.py
