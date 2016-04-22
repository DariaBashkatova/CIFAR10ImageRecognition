# TermProject
Tackling the CIFAR-10 Dataset Object Recognition Problem

Final Model Strategy:
- Train Multiple Convolutional Neural Networks
- Get the final hidden layer output of the multiple CNN's when each image (train and test) is inputted and store this
    "transformed" version of the data (a form of feature extraction using the CNN's)
    - the command "python run_cnn.py" will create, train, validate, and store a model and immediately after use
        this model to transform the CIFAR-10 train and test data and store this model as well (using cPickle)
- Use the stored, tranformed data, given by each of the multiple CNN's, to train a highly accurate SVM and
    print the results to a csv file stored in the data folder
    - the command "python run_extracted.py" will do the above
- We used 7 models of validation accuracies ranging from 82%-84.5% and received a test accuracy of 90.5%

Libraries Necessary to Run This Code:
- Python 2.7
- Numpy 1.11.0
- Scikit-Learn 0.18.dev0 (To run a scikit-learn NN (non-conv), which is not what our final model uses, otherwise scikit-learn 0.17 should work fine)
- Scikit-Image 0.11.3 (To run NN (non-conv) with HOG representation)
- Theano 8.0
- Lasagne 0.2.dev1
- Nolearn 0.6a0.dev0
- CUDA compatible GPU and CUDA setup (if running GPU to train Nolearn/Lasagne/Theano models for ~25x speedup)
- Scipy 0.17.0
- Matplotlib 1.5.1
- Pandas 0.18.0

How to Run This Code to produce final test labels:
- Download Code
- Download CIFAR-10 train and test data into a folder named "data"
- Download Pickled Model (If you are wanting to run code using pickled model/save training time)
- Run the command "python run_extracted.cnn"
- Labels will be stored in a csv file in the "data" folder

Methods Attempted:
- Naive Bayes (run_naive_bayes.py)
    - Gaussian Naive Bayes (28.33%)
    - Binomial Naive Bayes (27.28%)
    - Multinomial Naive Bayes (30.17%)
- Softmax Regression (run_softmax.py) (41.15%)
- One-vs-All Logistic Regression (run_ova_log.py) (41.56%)
- Neural Network (run_neural_network.py)
    - Single NN (52%)
    - Voting System of 5 NN's (55%)
    - Voting System of 5 NN's with HOG Representation (59%)
- Convolutional Neural Network (run_cnn.py)
    - Single CNN (83.0%)
    - 5 CNN's Voting (86.9%)
    - 5 CNN's Average Softmax Probability Output (87.9%)
    - 5 CNN's Average Log Softmax Probability Output (88.2%)
    - 5 CNN's Emsembled via SVM (run_extracted.py) (89.6%)

Credit Due To:
- Daniel Nouri and his impressive Nolearn Wrapper around Lasagne, as well as fantastic tutorial walkthrough,
    from which I used several functions, located in nn_utils.py (each one has a note in the docstrings)
    - http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
- Christian Perone and his Nolearn tutorial from which I took code on how to receive hidden layer output
    and how to visualize conv layer weights
    - http://blog.christianperone.com/2015/08/convolutional-neural-networks-and-feature-extraction-with-python/
- Bergstra and Bengio for their work ("Random Search for Hyperparameter Optimization") on showing that random sweep
    of the hyperparameter space is as effective, if not more effective than grid search in many/most cases
    (Heavily influenced how I parameter sweeped for NN's)
    - http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf
- He, Zhang, Ren, and Sun at Microsoft Research for their work "Deep Residual Learning for Image Recognition"
    for providing a complete CNN architecture and set of hyperparameters that worked extremely well and saved me
    and incredible amount of time with regards to sweeping the hyperparameter space
- Python, Numpy, Sci-kit Learn, Sci-kit Image, Theano, Lasagne, Nolearn, Scipy, Matplotlib, and Pandas libraries,
    all of which were used to produce these results