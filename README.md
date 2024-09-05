# MNIST classifier

Handwritten digit classifier for the MNIST database. These are composed
of 70000 28x28 pixel gray-scale images of handwritten digits divided into a 60000
training set and a 10000 test set.
Train the following classifiers on the dataset:
1. SVM using linear, polynomial of degree 2, and RBF kernels;
2. Random forest;
3. Naive Bayes classifier where each pixel is distributed according to a Beta distribution of parameters α, β:
   $$p(x_i | y) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} x_i^{\alpha - 1} (1 - x_i)^{\beta - 1}$$
5. k-NN.

You can use scikit-learn or any other library for SVM and random forests, but you
must implement the Naive Bayes and k-NN classifiers yourself.
Use 10-way cross-validation to optimize the parameters for each classifier.
Provide the code, the models on the training set, and the respective performances
in testing and 10-way cross-validation.
Explain the differences between the models, both in terms of classification performance and in terms of computational requirements (timings) in training and
prediction.

See the report for further details.
