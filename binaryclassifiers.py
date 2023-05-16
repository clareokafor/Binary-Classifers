""" Name:       Ogochukwu Jane Okafor
    Student ID: 201666459
"""

# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

# Loading the dataset information using the imported pandas
dataset = pd.read_csv('/dataset_assignment1.csv') 
print(dataset) # printing the dataset in order to visualise the data
print(dataset['class'].value_counts()) # Printing out the number of samples for each class in the dataset

# Plotting several histograms to visualize the dataset
X_feats = dataset.columns[:-1] # indicates data in all columns excluding the last column (class) which has the class label
colors = 'g' # setting the color of the dataset histogram to green
i = 0 
for x in X_feats: # for every feature in the dataset
  plt.hist(dataset[x], bins=10, color=colors[i]) # configuring the width and height of the histogram
  plt.title(f'The histogram showing {x}') # creating titles for all the features in the dataeset respectfully
  plt.xlabel(x) # labeling the x axis as the name of the x
  plt.ylabel('frequencies') # labeling the y axis to frequencies
  plt.show() # visualising the histogram
  i = (i+1) % len(colors) # let all the features from 1-9 have the assigned green color for their respective histograms.

# Printing out the statistical description of the features which include counts, mean, std, max and min values, for each class in the dataset.
for des in dataset['class'].unique(): # for every description existing and unique in the dataset
  print(f"Class {des} Statistical Description:\n{dataset[dataset['class']==des].describe()}\n") # this will print the class 1 and class 2 and their statistical descriptions.

# Splitting the data into X features and y class label
X_feat = dataset.drop('class', axis = 1) # class features will be all columns except for the the last column
y_label = dataset['class'] # class label will be the last column in the dataset

# Splitting the dataset into train data and test data based on the specified test size and random state.
train_data = 42 # setting the train data to a random state of 42 in order to split the data
test_data = 0.2 # assigning 0.2 proportion of the dataset intended for testing.

# Training the model using the training dataset
# Storing the train and test data for both features and class labels in four separate variables.
X_feat_tr, X_feat_te, y_label_tr, y_label_te = train_test_split(X_feat, y_label, random_state = train_data, test_size = test_data) 
 

"""***************************************************************************************************************************************************************************
STEP 5: CREATING KNN ALGORITHM """
# Creating a KNN classifier
knn_classifier = KNeighborsClassifier() # creating an instance of the KNeighborsClassifier class, which will be used to build the KNN model.

# Assigning hp_knn to be the dictionary holding the set of hyperparameters that will be tuned to find the best model
hp_knn = {'n_neighbors': list(range(5, 10)), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}

# Using GridSearchCV to tune the hyperparameters of the KNN classifier
best_k = [] # an empty list to hold the best k values.
scorings = ['accuracy','precision', 'recall', 'f1'] # holding the four required evaluation metrices in a list.
for s in scorings: # for any evaluation metrice in the scorings list
  knn_grid = GridSearchCV(knn_classifier, hp_knn, scoring=s) # using the GridSearchCV class to iterate over all possible combinations of hyperparameters and find the best parameters for each metric.
  knn_grid.fit(X_feat_tr, y_label_tr) # fitting the knn classifier with a grid search to find the best hyperparameters.
  knn_classifier = knn_grid.best_estimator_ # the best estimator is re-assigned to the knn_classifier 
  best_k.append(knn_grid.best_params_['n_neighbors']) # the best number of neighbors are appended to best_k.
print(f"The best K Values for KNN are: {best_k}") # Printing the best hyperparameters for the KNN classifier

# Using evaluation metrics to pick up a model that gives you the best result on the validation dataset
# Slicing the k values list to get the models
knn_models = best_k[:-1] # assigning knn_models to represent the models after slicing the best_k list.
for k in knn_models: # for every k values in this list except for the last column since the object in that column appeared before.
  knn_classifier = KNeighborsClassifier(n_neighbors=k) # using the k neighbors to classify the model
  knn_classifier.fit(X_feat_tr, y_label_tr) # fitting the knn classifier with the features and class labels of the train data for validation
  kpred_test = knn_classifier.predict(X_feat_te) # predicting for knn classifier using the fetaures of the test data

# Calculating the evaluation metrics for the test data using the class label, prediction and the average
  p_k = precision_score(y_label_te, kpred_test, average='macro') # precision score for the test data of any model in knn
  r_k = recall_score(y_label_te, kpred_test, average='macro') # recall score for the test data of any model in knn
  f1_k = f1_score(y_label_te, kpred_test, average='macro') #f1 score for the test data of any model in knn
  a_k = accuracy_score(y_label_te, kpred_test) # acccuracy score for the test data of any model in knn
  k_c = confusion_matrix(y_label_te, kpred_test) # confusion matrix for the test data of any model in knn

# Printing the evaluation metrics and the confusion matrix for the test data
  print('KNN Model at K = ', k) # will print the K Nearest Neighbor model at each K 
  print('Precision:', p_k) # will print the Precision for each model in KNN
  print('Recall:', r_k) # will print the Recall for each model in KNN
  print('F1 Score:', f1_k) # will print the F1 Score for each model in KNNl
  print('Accuracy:', a_k) # will print the Accuracy for each model in KNN
  print('Confusion Matrix: \n ', k_c) # will print the Confusion Matrix for each model in KNN

# Plotting a Bar Chart to summarise the information after evaluation
  classes = dataset['class'].unique() # for all classes unique in the dataset
  evaluations = ['precision', 'recall', 'f1', 'accuracy'] # the evaluation metrices
  scores = [] # an empty list to hold scores for the four evaluation metrices
  for y in classes: # for every class unique in dataset
    true_class = (y_label_te == y) # the true class is the class label in the dataset
    pred_class = (kpred_test== y) # assigned pred_class to be the predicted class for the class label
    p_k = precision_score(true_class, pred_class) # determining precision score using true class and predicted class
    r_k = recall_score(true_class, pred_class) # determining recall score using true class and predicted class
    f1_k = f1_score(true_class, pred_class) # determining f1 score using true class and predicted class
    a_k = accuracy_score(true_class, pred_class) # determining accuracy score using true class and predicted class
    scores.append([p_k, r_k, f1_k, a_k]) # appending the precision, recall, f1 and accuracy scores to the scores list

  #  Initializing a plot with a figure size of 8 by 6.
  fig, ax = plt.subplots(figsize=(8,6)) 
  id = np.arange(len(classes)) # creating an array of integers from 0 to the length of classes.
  wdt = 0.2 # setting the width of each bar in the plot.
  opacity = 0.8 # setting the opacity/transparency of each bar in the plot.
  cl = ['c', 'm', 'y', 'k'] # an array of colors for each evaluation metric.

  for i in range(len(evaluations)): #  iterating through each evaluation metric
    ax.bar(id + i * wdt, [score[i] for score in scores], wdt, alpha=opacity, color=cl[i], label=evaluations[i]) # plotting a bar for each class in the test data using the specified color.
  ax.set_xlabel('Class') # labelling the x axis as class
  ax.set_ylabel('Score')# labelling the y axis s score
  ax.set_title(f'The KNN Evaluation Metrics for Each Class in the Test Data (K={k})') # titling the plot including the value of K used in the KNN model.
  ax.set_xticks(id+wdt) # setting the x-axis tick positions
  ax.set_xticklabels(classes) # setting to the center of each bar and labeled with the class names.
  ax.legend() # identifying each evaluation metric.
  plt.show() # displaying the bar chart for KNN using the plt.show() function.


"""***************************************************************************************************************************************************************************
STEP 5: CREATING A DECISION TREE ALGORITHM"""

# Assigning hp to be the dictionary holding the set of hyperparameters that will be tuned to find the best model
hp = {'max_depth': list(range(3, 10)), 'min_samples_split': list(range(5, 50))}

# Creating the decision tree classifier
dtree_classifier = DecisionTreeClassifier()

# Using GridSearchCV to tune the hyperparameters of the Decision Tree Algorithm.
dt_best_m = [] # Creating a list to store the best max_depth for each scoring metric
dt_scorings = ['accuracy','precision', 'recall', 'f1'] # holding the four required evaluation metrices in a list.

# Iterating over the list of scoring evaluation metrics
for s in dt_scorings: # for any evaluation metrice in the scorings list

    # Use GridSearchCV to tune the hyperparameters of the decision tree classifier
    dtree_grid = GridSearchCV(dtree_classifier, hp, scoring=s) # using the GridSearchCV class to iterate over all possible combinations of hyperparameters and find the best parameters for each metric.
    dtree_grid.fit(X_feat_tr, y_label_tr) # fitting the decision tree classifier with a grid search to find the best hyperparameters.
    best_params = dtree_grid.best_params_ # the best parameter is re-assigned to the dtree_classifier 
    dt_best_m.append(best_params['max_depth']) # the best max depths are appended to best_k.
print(f"The best max depth for each scoring metric are: {dt_best_m}") # Printing the best hyperparameters for the Decision Tree Classifier

# Using evaluation metrics to pick up a model that gives you the best result on the validation dataset
dtree_models = dt_best_m[:-2] # assigning dtree_models to represent the models after slicing the max_depths list
for m in dtree_models: # for every max depth in this list except for the last two columns since these objects have appeared before.
  dtree_classifier = DecisionTreeClassifier(max_depth=m) # using the max depths to classify the model
  dtree_classifier.fit(X_feat_tr, y_label_tr) # fitting the decision tree classifier with the features and class labels of the train data for validation
  dpred_test = dtree_classifier.predict(X_feat_te) # predicting for decision tree classifier using the fetaures of the test data

# Calculating the evaluation metrics for the test data using the class label, prediction and the average
  p_dt = precision_score(y_label_te, dpred_test, average='macro') # precision score for the test data of any model in decision tree
  r_dt = recall_score(y_label_te, dpred_test, average='macro') # recall score for the test data of any model in decision tree
  f1_dt = f1_score(y_label_te, dpred_test, average='macro') # f1 score for the test data of any model in decision tree
  a_dt = accuracy_score(y_label_te, dpred_test) # accuracy score for the test data of any model in decision tree
  cm_dt = confusion_matrix(y_label_te, dpred_test) # confusion matrix for the test data of any model in decision tree

# Printing the evaluation metrics and the confusion matrix for the test data
  print('Decision Tree Model at M = ', m) # will print the Decision Tree Model at each Max Depth
  print('Precision:', p_dt) # will print the Precision for each model in Decision Tree
  print('Recall:', r_dt) # will print the Recall for each model in Decision Tree
  print('F1 Score:', f1_dt) # will print the F1 Score for each model in Decision Tree
  print('Accuracy:', a_dt) # will print the Accuracy for each model in Decision Tree
  print('Decision Tree Confusion Matrix: \n ', cm_dt) # will print the Confusion Matrix for each model in Decision Tree

# Plotting A Decision Tree of the Decison Tree Classifier"
plt.figure(figsize=(20,10)) # initializing a plot with a figure size of 20 by 10.
plot_tree(dtree_classifier, filled=True, class_names=['0', '1']) # visualizing the decision tree model and makes predictions through splits and decision paths taken by the tree.
plt.show() # displaying the decision tree plot for the Decision Tree Algorithm using the plt.show() function.


"""***************************************************************************************************************************************************************************
STEP 5: CREATING LOGISTIC REGRESSION ALGORITHM"""
# Creating a logistic regression classifier
logreg_classifier = LogisticRegression()

# Defining hyperparameters for logistic regression
hp_logreg = {'C': [0.001, 0.01, 0.1, 1.0, 10.0], 'solver': ['lbfgs', 'liblinear'], 'max_iter': [1000]}

# Using GridSearchCV to tune the hyperparameters of the Logistic Regression Classifier
best_c = []
lg_scorings = ['accuracy', 'precision', 'recall', 'f1'] 
for s in lg_scorings: # for any evaluation metrice in the scorings list
    # Define the KFold cross-validator with 5 splits
    logreg_grid = GridSearchCV(logreg_classifier, hp_logreg, scoring=s) # using the GridSearchCV class to iterate over all possible combinations of hyperparameters and find the best parameters for each metric.
    logreg_grid.fit(X_feat_tr, y_label_tr) # fitting the logistic regression classifier with a grid search to find the best hyperparameters.
    logreg_classifier = logreg_grid.best_estimator_ # the best estimateor is re-assigned to the logreg_classifier 
    best_c.append(logreg_grid.best_params_['C']) # the best regularisation parameters are appended to best_k.
print(f"The best Regularisation Parameter for logistic regression are: {best_c}") # Printing the best hyperparameters for logistic regression

# Using evaluation metrics to pick up a model that gives the best result on the validation dataset
log_models = best_c[:-2] # assigning log_models to represent the models after slicing the regularisation parameters list.
for c in log_models: # for every regularisation parameter in this list except for the last two columns since these objects have appeared before.
    logreg_classifier = LogisticRegression(C=c) # using the regularisation parameters to classify the model
    logreg_classifier.fit(X_feat_tr, y_label_tr) # fitting the logistic regression classifier with the features and class labels of the train data for validation
    lrpred_test = logreg_classifier.predict(X_feat_te) # predicting for logistic regression classifier using the fetaures of the test data

    # Calculating the evaluation metrics for the test data using the class label, prediction and the average
    p_lr = precision_score(y_label_te, lrpred_test, average='macro') # precision score for the test data of any model in logistic regression
    r_lr = recall_score(y_label_te, lrpred_test, average='macro') # reacll score for the test data of any model in logistic regression
    f1_lr = f1_score(y_label_te, lrpred_test, average='macro') # f1 score for the test data of any model in logistic regression
    a_lr = accuracy_score(y_label_te, lrpred_test) # accuracy score for the test data of any model in logistic regression
    lr_c = confusion_matrix(y_label_te, lrpred_test) # confusion matrix for the test data of any model in logistic regression

    # Printing the evaluation metrics and the confusion matrix for the test data
    print('Logistic Regression Model at C =', c) # will print the Logistic Regression Model at each Regularisation Parameter
    print('Precision:', p_lr) # will print the precision for each model in Logistic Regression
    print('Recall:', r_lr) # will print the recall for each model in Logistic Regression
    print('F1 Score:', f1_lr) # will print the f1 score for each model in Logistic Regression
    print('Accuracy:', a_lr) # will print the accuracy for each model in Logistic Regression
    print('Logistic Regression Confusion Matrix: \n', lr_c) # will print the Confusion Matrix for each model in Logistic Regression

# Plotting for the ROC (Receiver Operating Characteristic) Curve
# Predicting the probabilities of the target class using the logistic regression model
lrpred_prob = logreg_classifier.predict_proba(X_feat_te)[:,-1]

# Calculating the FPR, TPR and threshold values
fpr, tpr, thresholds = roc_curve(y_label_te, lrpred_prob)

# Plotting the ROC curve for Logistic Regression
plt.plot(fpr, tpr, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--') # plotting a the diagonal line to represent the ROC curve of the classifier.
plt.xlabel('False Positives') # sets the label for the x-axis.
plt.ylabel('True Positives') # sets the label for the y-axis.
plt.title('Receiver Operating Characteristic (ROC) Curve') # sets the title of the plot.
plt.legend() # will display the label of each line.
plt.show() # will display the ROC plot.

k_values = best_k[:-1] # lists the best k values for k-NN classifier except the last value.
m_values = dt_best_m[:-2] # lists the best max_depth values for decision tree classifier except the last two values.
c_values = best_c[:-2] # lists the best C values for logistic regression classifier, except the last two values.

"""***************************************************************************************************************************************************************************
"""
# Defining a Plot Function to Compute the Confusion Matrix of the Three Algorithms
def PlotConfusionMatrix(model, X_feat_te, y_label_te, algorithm, parameter):
    """
   Plot Confusion Matrix  Function:  
              This function will plot the diagrams to visualise the confusion matrices for the best models under the three methods being tested.
              The confusion matrix is then calculated after the labels are predicted using the test features and the provided model.
              It displays the plot and plots the confusion matrix using seaborn heatmap.
    
    Parameters:
          1.  model:      for knn classifier, dtree classifer and log classifier.
          2.  X_feate_te: test features in knn, dtree and log. 
          3.  y_label_te: test class labels in knn, dtree and log.
          4.  algorithm:  k, m and c for knn, dtree and log respectively.
    
    Returns:
            None
    """
    pred = model.predict(X_feat_te) # predicting a class using the x features of the test data for each method.
    cm = confusion_matrix(y_label_te, pred) # determining the confusion matrix through the class label and predicted class for each method.
    plt.figure(figsize=(8, 6)) # initializing a plot with a figure size of 8 by 6.
    if algorithm == 'KNN': # if algorithm is knn
        cmap = 'RdPu'# setting the color map 
        title = f'Confusion Matrix for {algorithm} (K Value = {k})' # the title for the confusion matrix for each best k value in knn.
    elif algorithm == 'Decision Tree': # if algorithm is dtree
        cmap = 'GnBu' # setting the color map 
        title = f'Confusion Matrix for {algorithm} (Max Depth = {m})'# the title for the confusion matrix for each best max depts in dtree.
    elif algorithm == 'Logistic Regression': # if algorithm is log
        cmap = 'YlOrRd' # setting the color map 
        title = f'Confusion Matrix for {algorithm} (Regularisation Parameter = {c})' # the title for the confusion matrix for each c in log.
    sns.heatmap(cm, annot=True, cmap=cmap, fmt='g') # using seaborn's heatmap to plot the confusion matrix 
    plt.title(title) # titling the plot 
    plt.xlabel('Predicted Class') # labelling the x axis as predicted class
    plt.ylabel('True Class') # labelling the y axis as true class
    plt.show() # displaying the plot.

for k in k_values: # iterating over a list of values k_values.
  knn_classifier = KNeighborsClassifier(n_neighbors=k) # initializing the KNeighbors Classifier object with the current value of k.
  knn_classifier.fit(X_feat_tr, y_label_tr) # calling on the KNeighbors Classifier object to train the model on the training data.
  PlotConfusionMatrix(knn_classifier, X_feat_te, y_label_te, 'KNN', k) # calling the PlotConfusionMatrix function is to generate a confusion matrix plot for KNN.

for m in m_values:# iterating over a list of values km_values.
  dtree_classifier = DecisionTreeClassifier(max_depth=m) # initializing the DecisionTree Classifier object with the current value of m.
  dtree_classifier.fit(X_feat_tr, y_label_tr) # calling on the DecisionTree Classifier object to train the model on the training data.
  PlotConfusionMatrix(dtree_classifier, X_feat_te, y_label_te, 'Decision Tree', m) # calling the PlotConfusionMatrix function is to generate a confusion matrix plot for Decision Tree.

for c in c_values: # iterating over a list of values c_values.
  logreg_classifier = LogisticRegression(C=c) # initializing the LogisticRegression Classifier object with the current value of k
  logreg_classifier.fit(X_feat_tr, y_label_tr) # calling on the LogisticRegression Classifier object to train the model on the training data.
  PlotConfusionMatrix(logreg_classifier, X_feat_te, y_label_te, 'Logistic Regression', c) # calling the PlotConfusionMatrix function is to generate a confusion matrix plot for Logistic Regression.


# Designinating a function to plot a classification report (Bar Chart) for the three methods
def PlotClassificationReport(X_feat_te, y_label_te, knn_classifier, dtree_classifier, logreg_classifier):
    """
    PlotClassificationReport Function:
              This function plots a bar chart to display the performance of each model using the classification_report metric for KNN, Decision Tree and Logistic Regression methods.

    Parameters:
            1. X_feat_te: array-like for test set X features.
            2. y_label_te: array-like for test est set classlabels.
            3. knn_classifier:  object-
                                - KNN classifier object.
                                - Decision Tree classifier object.
                                - Logistic Regression classifier object.

    Returns:
            None
    """
    # Get predictions for each model
    kpred_test = knn_classifier.predict(X_feat_te) # predicting test for knn
    dpred_test = dtree_classifier.predict(X_feat_te) # predicting test for dtree
    lrpred_test = logreg_classifier.predict(X_feat_te) # predicting test for log

    # Calculate classification report for each model
    cr_knn = classification_report(y_label_te, kpred_test, output_dict=True) # calculating for knn
    cr_dt = classification_report(y_label_te, dpred_test, output_dict=True) # calculating for dtree
    cr_lr = classification_report(y_label_te, lrpred_test, output_dict=True) # calculating for log

    # Extract accuracy, precision, recall, and F1 score for each model
    accuracy = [cr_knn['accuracy'], cr_dt['accuracy'], cr_lr['accuracy']] # extracting for accuracy
    precision = [cr_knn['weighted avg']['precision'], cr_dt['weighted avg']['precision'], cr_lr['weighted avg']['precision']] # extracting for precision
    recall = [cr_knn['weighted avg']['recall'], cr_dt['weighted avg']['recall'], cr_lr['weighted avg']['recall']] # extracting for recall
    f1_score = [cr_knn['weighted avg']['f1-score'], cr_dt['weighted avg']['f1-score'], cr_lr['weighted avg']['f1-score']] # extracting for f1 score

    # Create a bar chart to display the performance of each model
    fig, ax = plt.subplots() # creating a figure and axes objects for the chart.
    model_names = ['KNN', 'Decision Tree', 'Logistic Regression'] # defining the names of the models that will be displayed on the x-axis. 
    x_pos = [i for i, _ in enumerate(model_names)] # defining a list comprehension that assigns a position on the x-axis for each model name.
    ax.bar(x_pos, accuracy, label='Accuracy', width=0.2) # creating four bars on the chart for each model. 
    ax.bar([i+0.2 for i in x_pos], precision, label='Precision', width=0.2) # positioning the on the x-axis to precision
    ax.bar([i+0.4 for i in x_pos], recall, label='Recall', width=0.2) # positioning the on the x-axis to recall
    ax.bar([i+0.6 for i in x_pos], f1_score, label='F1 Score', width=0.2) # positioning the on the x-axis to f1 score
    ax.set_xlabel('Model') # labelling the x axis to model
    ax.set_ylabel('Score') # labelling the x axis to score
    ax.set_xticks([i+0.3 for i in x_pos]) # setting the tick positions for the x-axis
    ax.set_xticklabels(model_names) # setting the tick labels for the x-axis
    ax.legend() # adding a legend to the chart
    plt.show()# displaying the chart

# calling the plot classification report function to compute the plots
PlotClassificationReport(X_feat_te, y_label_te, knn_classifier, dtree_classifier, logreg_classifier)