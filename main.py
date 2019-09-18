import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"

def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(X[:,0],X[:,1],c=y,cmap='RdBu')
    plt.savefig('train_features.png')
    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot the sigmoid model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].

    Returns:
        No return. Save the plot to 'train_result_sigmoid.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    import matplotlib.pyplot as plt
    _decision = -(W[0]+W[1]*X[:,0])/W[2]
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y,cmap='RdBu')
    plt.plot(X[:, 0], _decision,'k-')
    plt.savefig('train_result_sigmoid.png')
    ### END YOUR CODE

def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].

    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    import matplotlib.pyplot as plt
    _decision1 = -(W[0,0] + W[1,0] * X[:, 0]) / W[2,0]
    _decision2 = -(W[0,1] + W[1,1] * X[:, 0]) / W[2,1]
    _decision3 = -(W[0,2] + W[1,2] * X[:, 0]) / W[2,2]

    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Accent')
    plt.plot(X[:, 0], _decision1, 'k-')
    plt.plot(X[:, 0], _decision2, 'r-')
    plt.plot(X[:, 0], _decision3, 'g-')
    plt.ylim([-1,1])
    plt.savefig('train_result_softmax.png')
    ### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.

    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    data_shape= train_y.shape[0]

    #    # Visualize training data.
    visualize_features(train_X[:, 1:3], train_y)


    # ------------Logistic Regression Sigmoid Case------------

    ##### Check GD, SGD, BGD
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=100)

    logisticR_classifier.fit_GD(train_X, train_y)
    print('GD')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    #preds_proba = logisticR_classifier.predict_proba(train_X)

    logisticR_classifier.fit_SGD(train_X, train_y)
    print('SGD')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, data_shape)
    print('BGD')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    '''
    logisticR_classifier.fit_BGD(train_X, train_y, 1)
    print('BGD, size = 1')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 10)
    print('BGD, size = 10')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    '''

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    '''
    logisticR_classifier.fit_BGD(train_X, train_y, 100)
    print('BGD, size = 100')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 500)
    print('BGD, size = 500')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, 1000)
    print('BGD, size = 1000')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))

    logisticR_classifier.fit_BGD(train_X, train_y, train_X.shape[0])
    print('BGD, size = {}'.format(train_X.shape[0]))
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    '''
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    ### YOUR CODE HERE
    visualize_result(train_X[:, 1:3], train_y, logisticR_classifier.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    print('test (validation) accuracy')
    print(logisticR_classifier.score(valid_X, valid_y))
    ### END YOUR CODE


    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  BGD for multiclass Logistic Regression
    '''
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 10)
    print('0.5,10')
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 100)
    print('0.5, 100')
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 1000)
    print('0.5, 1000')
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))

    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, train_X.shape[0])
    print('0.5, full_range')
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))
    
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.9, max_iter=100,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, train_X.shape[0])
    print('0.9, full_range')
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))
    '''
    '''
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=2, max_iter=200,  k= 3)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, train_X.shape[0])
    print('Hyper Parameters; 2,200, full_range')
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    visualize_result_multi(train_X[:, 1:3], train_y, logisticR_classifier_multiclass.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    print('test (validation) accuracy')
    print(logisticR_classifier_multiclass.score(valid_X, valid_y))
    ### END YOUR CODE
    '''

    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0

    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter=10000, k=2)
    logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 100)
    print('Softmax, Hyper Parameters; 0.5,10000 iter,100 batch full_range')
    print(logisticR_classifier_multiclass.get_params())
    print(logisticR_classifier_multiclass.score(train_X, train_y))
    ### END YOUR CODE

    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    logisticR_classifier = logistic_regression(learning_rate=0.5, max_iter=10000)
    logisticR_classifier.fit_BGD(train_X, train_y,100)
    print('BGD')
    print(logisticR_classifier.get_params())
    print(logisticR_classifier.score(train_X, train_y))
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    '''
    Explore the training of these two classifiers and monitor the graidents/weights for each step. 
    Hint: First, set two learning rates the same, check the graidents/weights for the first batch in the first epoch. What are the relationships between these two models? 
    Then, for what leaning rates, we can obtain w_1-w_2= w for all training steps so that these two models are equivalent for each training step. 
    '''
    ### YOUR CODE HERE
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0
    print('-----------softmax -----------')
    for iters in [1000,2000,3000,4000]:
        logisticR_classifier_multiclass = logistic_regression_multiclass(learning_rate=0.5, max_iter = iters, k=2)
        logisticR_classifier_multiclass.fit_BGD(train_X, train_y, 100)
        W_out = logisticR_classifier_multiclass.get_params()
        W1_2 = W_out[:,0]-W_out[:,1]
        print(W1_2)


    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    #####       set lables to -1 and 1 for sigmoid classifer
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1

    print('-----------logistic -----------')
    for iters in [1000,2000,3000,4000]:
        logisticR_classifier = logistic_regression(learning_rate=1, max_iter=iters)
        logisticR_classifier.fit_BGD(train_X, train_y,100)
        print(logisticR_classifier.get_params())
    ### END YOUR CODE

# ------------End------------


if __name__ == '__main__':
    main()
    
    
