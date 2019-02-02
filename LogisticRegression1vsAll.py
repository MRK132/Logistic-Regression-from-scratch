import numpy as np # used for linear algebra
import pandas as pd # used for loading and saving .csv datasets and results of predictions
import matplotlib.pyplot as plt # used to plot confusion matrix
import seaborn as sns # used to plot confusion matrix
from sklearn.metrics import confusion_matrix # used to plot confusion matrix
from sklearn.model_selection import train_test_split # used to split data between training and testing

av_Array = []    # will be populated with the accuracy of 10 runs of the program to compute an overall average accuracy
runs = 10
for i in range(runs): #loop through main Logistic Regression implementation 10 times


    owls = pd.read_csv('C:/Users/Mark/Desktop/4BP/Machine Learning/owls.csv')
    print(owls.head())  #to show the dataset is correctly formatted, print out the first 5 rows of data


        #  Data Preprocessing Stage:
    ################################################################################################
    OwlType = ['LongEaredOwl', 'SnowyOwl', 'BarnOwl']   #class labels

    numRows = owls.shape[0]   # 135 rows of data

    numFeatures = 4  # owl features
    numClasses = 3   # owl classes

    X = np.zeros((numRows,numFeatures))    # create a matrix of zeroes that is 135 rows and 4 columns in shape that will house the data
    print("Array X before population of data: ")

    print(X)
    print('\n')

    X[:,0] = owls['body-length'].values #populates the empty dataset with the values of our features
    X[:,1] = owls['wing-length'].values
    X[:,2] = owls['body-width'].values
    X[:,3] = owls['wing-width'].values

    print("Array X after population of data: ")
    print(X)
    print('\n')

    y = owls['type'].values     #class labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33333, random_state = None) # split into 2 thirds train, 1 third test, randomise each time

    def sigmoid(z):         #sigmoid function
        return 1.0 / (1 + np.exp(-z))
    theta = np.zeros(numFeatures)


    def gradient_Descent(X, y, learningRate, iterations):       # implementation of the eqns to find parameters to minimze loss with gradient descent
        numFeatures = X.shape[1]
        theta = np.zeros(numFeatures)          # initialization of theta

        for i in range(iterations):
            preds = sigmoid(X.dot(theta))             # compute predictions based on the current hypothesis
            m = len(y)
            error = preds - y

            gradient = X.transpose().dot(error)  # Here we are computing the partial derivatives of the cost

            theta -= learningRate * (gradient * (1/m)) # gradient descent algorithm: update rule

        return theta


    # Train classifiers: One vs all
#########################################################################################################################################################################################


    size_Theta = np.zeros((numClasses, numFeatures))    #initial size for theta
    t = 0
    for owl in (OwlType):

    #For each owl type, create the temp_y_train labels
    #Assign values to 1 when values in temp_y_train equal the current owl type labels, assign other values to 0

        temp_y_train = np.array(y_train == owl, dtype = int)
        optimal_Theta = gradient_Descent(X_train, temp_y_train, 0.01, 10000)     #Perform gradient descent to find best parameters
        size_Theta[t] = optimal_Theta # populated with optimal parameters
        t += 1

######################################################################################################################################################################################
    # Make predictions:

    prob_owl_type = sigmoid(X_test.dot(size_Theta.T)) #probability for each owl
    predictions = [OwlType[np.argmax(prob_owl_type[i, :])] for i in range(X_test.shape[0])] # most confident probability corresponds to the assigned predicted class

    print("\n")
    print("GROUND TRUTH LABELS:")
    print(y_test)
    print("\n")
    print("PREDICTIONS")
    print(predictions)
    print("\n")

    acc_Classify=np.mean(predictions==y_test)       # Compute accuracy of classifier
    print( "Classification Accuracy of this run: ", acc_Classify * 100 , '%' )
    print("\n")
    av_Array.append(acc_Classify)


print("CLASSIFICATION ACCURACIES FOR ALL 10 RUNS: ")
print(av_Array)
print("\n")
def AverageOverall(av_Array):       #function to compute average of all runs
    return np.mean(av_Array)

overall_Av = AverageOverall(av_Array)
print("Overall average of classification accuracy is  ", overall_Av * 100, '%')

                                    #Output to .csv for iteration 10
df = pd.DataFrame({"Ground_truth_labels" : y_test, "predictions" : predictions})
df.to_csv('Ground_truth_labels_VS_Predictions.csv')

                                    #Confusion Matrix for iteration 10
confusion_Matrix = confusion_matrix(y_test, predictions, labels = OwlType)

sns.heatmap(confusion_Matrix, annot = True, xticklabels = OwlType, yticklabels = OwlType);
plt.title('Logistic Regression \nIteration 10 Classification Accuracy: {0:.4f}'.format(acc_Classify * 100, '%'))
plt.xlabel('Predicted class label')
plt.ylabel('Ground truth label')
plt.show()
