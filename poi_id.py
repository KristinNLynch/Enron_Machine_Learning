#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/python

#USE PYTHON 3
#TESTER FILE AND OTHER FILES UPDATED TO PYTHON 3
import sys
import pickle
import csv
import numpy as np



from time import time

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import fbeta_score , accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier




#sys.path.append("../tools/")
#sys.path.append("C:/ud120-projects/tools/")

with open("final_project_dataset.pkl", "rb") as f:
     data_dict = pickle.load(f)


from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

#Tester file was converted to python 3


# In[2]:


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### poi_id

features_list = ['poi','salary' ,'to_messages','deferral_payments','total_payments','loan_advances','bonus',
'deferred_income',
'expenses',
'from_poi_to_this_person',
'exercised_stock_options',
'from_messages',
'other',
'from_this_person_to_poi',
'long_term_incentive',
'shared_receipt_with_poi',
'restricted_stock']
print(len(features_list))


# In[3]:


# Task 2: Remove outliers
 


# In[4]:


# add to poi_id
#Remove these outliers from the dictionary.
#Removed due to too much missing data in the record. 
data_dict.pop('URQUHART JOHN A', None)
#Removed this as it is not a person.
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)
data_dict.pop('TOTAL', None)
len(data_dict)


# In[5]:


# add to poi_id
#Removed due to too much missing data in the record and wrong totals.
data_dict.pop('LAY KENNETH L', None)

len(data_dict)


# In[6]:


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# Add to poi_id
#Check the proportion of messages to a person of interest
def poi_message_ratio(data_dict, features_list):
    '''Add calculated field to data dictionary
    data_dict- enter the data dictionary
    features_list enter desired features'''
    
    for data in data_dict:
        employee_data = data_dict[data]
        print(data)
        
   
        from_message = employee_data['from_messages']
        to_poi = employee_data['from_this_person_to_poi'] 
            
        print("From Messages:", from_message)
        print("To POI", to_poi)
        if to_poi == 'NaN':
            to_poi = 0
        else:
            to_poi = int(to_poi)
        if from_message == 'NaN':
            employee_data['POI_MESSAGE_RATIO'] = 0
        else:
            to_poi = int(to_poi)
            employee_data['POI_MESSAGE_RATIO'] = to_poi / from_message
            print("Ratio", employee_data['POI_MESSAGE_RATIO'])
            
          
    features_list += ['POI_MESSAGE_RATIO']




# In[7]:


# Add to poi_id
#New feature
def total_compensation(data_dict, features_list):
    '''Add calculated field to data dictionary
    data_dict- enter the data dictionary
    features_list enter desired features'''
    field = ['salary', 'bonus','expenses']
    for data in data_dict:
        employee_data = data_dict[data]
        
      
        salary = employee_data['salary']
        bonus = employee_data['bonus']
        expenses = employee_data['expenses']
        
        if salary == 'NaN':
            salary = 0
        else: 
            salary = int(salary)
        if bonus == 'NaN':
            bonus  = 0
        else:
            bonus = int(bonus)
        if expenses == 'NaN':
            expenses = 0
        else:
            expenses = int(expenses)
            
        employee_data['TOTAL_COMPENSATION'] = salary + bonus + expenses
        
        print(data, 'TOTAL COMP', employee_data['TOTAL_COMPENSATION'])
           
            
          
    features_list += ['TOTAL_COMPENSATION']


# In[8]:


poi_message_ratio(data_dict, features_list)


# In[9]:


total_compensation(data_dict, features_list)


# In[10]:


### Extract features and labels from dataset for local testing
# add to poi_id
my_dataset = data_dict

data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)


# In[11]:


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
#add to poi_id


# In[12]:


#Scale the data
# add to poi_id
scale = preprocessing.MinMaxScaler()
features = scale.fit_transform(features)


# In[13]:


#Reapply PCA with 15 components:
#add to poi_id
components = 15
pca = PCA(components)
features = pca.fit_transform(features)


# In[14]:


# Provided to give you a starting point. Try a variety of classifiers.


# In[15]:


# Example starting point. Try investigating other evaluation techniques!
#This was changed to python 3 since cross validation is depreciated.
#add to poi_id
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)



# In[16]:


#Taken and adapted from my Finding Donors project
# add to poi_id
def train_predict(learner,  X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
  
    results = {}

    
    # Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:43], y_train[:43])
    end = time() # Get end time
    
    
    
    
    # Calculate the training time
    results['train_time'] = end - start
    
    
        
    # Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:43])
    end = time() # Get end time
    
    # Calculate the total prediction time
    pred_time = end - start
            
    # Compute accuracy on the first 300 training samples which is y_train[:300]
    acc_train = accuracy_score(y_train[:43], predictions_train)
        
    # Compute accuracy on test set using accuracy_score()
    acc_test = accuracy_score(y_test, predictions_test)
    
    # Compute F-score on the the first 300 training samples using fbeta_score()
    f_train = fbeta_score(y_train[:43], predictions_train, beta = .5)
    
    
    # Compute F-score on the test set which is y_test
    f_test = fbeta_score(y_test, predictions_test, beta = .5)
    
  
    # Success
    print("{} trained on 43 samples.".format(learner.__class__.__name__))
        
    # Return the results
    #print(results)
    return pred_time, acc_train, acc_test, f_train, f_test


# In[17]:


#Initialize the models
#Taken and Adapted from my Finding Donors project.
#add to poi_id
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

clf_A = GaussianNB()
clf_B = LogisticRegression()
clf_C = SGDClassifier()
clf_D = RandomForestClassifier()
clf_E = DecisionTreeClassifier()




# Collect results on the learners
results = {}
pred_list = []
acc_list = []
acc_test_list = []
f_train_list = []
f_test_list = []
model_list = ['GaussianNB', 'LogisticRegression', 'SGDClassifier', 'RandomForestClassifier','AdaBoost']
for clfs in [clf_A, clf_B, clf_C, clf_D, clf_E]:
    clf_name = clfs.__class__.__name__
    results[clf_name] = {}
    
    
   
    pred_time, acc_train, acc_test, f_train, f_test= train_predict(clfs, features_train, labels_train, features_test, labels_test)
    pred_list.append(pred_time)
    acc_list.append(acc_train)
    acc_test_list.append(acc_test)
    f_train_list.append(f_train)
    f_test_list.append(f_test)


# In[18]:


#add to poi_id
clf_list =  [clf_A,  clf_C, clf_D, clf_E]
for clf in clf_list:
    test_classifier(clf, my_dataset, features_list)


# In[19]:


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
#add to poi_id


# In[20]:


# Adapted from my finding donors project
#add to poi_id
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

#Initialize the classifier
clf_LR = LogisticRegression(random_state = 55, max_iter = 100)




parameters = {'C': [.24,.39 , .4, .41, .5], 'solver':['liblinear', 'newton-cg','lbfgs','sag','saga'], 
              'fit_intercept':[True, False] }




#  Make an fbeta_score scoring object using make_scorer()
scorer = make_scorer(fbeta_score,beta = .5)

# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
grid_obj = GridSearchCV(clf_LR, parameters, scoring = scorer, cv = 4)

# Fit the grid search object to the training data and find the optimal parameters using fit()
grid_fit = grid_obj.fit(features_train, labels_train)

# Get the estimator
best_clf = grid_fit.best_estimator_
print(best_clf)
# Make predictions using the unoptimized and model
predictions = (clf_LR.fit(features_train, labels_train)).predict(features_test)
best_predictions = best_clf.predict(features_test)

# Report the before-and-afterscores
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(labels_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(labels_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(labels_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(labels_test, best_predictions, beta = 0.5)))


# In[21]:


#Initialize the models
#Taken and Adapted from my Finding Donors project.
#add to poi_id
clf_LR = LogisticRegression(C=0.5, fit_intercept=False, random_state=55,solver='liblinear')
clf_RFC = RandomForestClassifier(bootstrap =  False, class_weight = 'balanced', random_state = 55)
clf_DT = DecisionTreeClassifier(class_weight = 'balanced')

print(clf_DT.get_params())


modified_list = ['LogisticRegression','RandomForestClassifier']
for clfs in [ clf_LR, clf_RFC]:
    clf_name = clfs.__class__.__name__
    results[clf_name] = {}
    
    
   
    pred_time, acc_train, acc_test, f_train, f_test= train_predict(clfs, features_train, labels_train, features_test, labels_test)
    pred_list.append(pred_time)
    acc_list.append(acc_train)
    acc_test_list.append(acc_test)
    f_train_list.append(f_train)
    f_test_list.append(f_test)


# In[22]:


## Test the model without one of the created fields - the poi message ratio
features_list2 = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'deferred_income', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 
                   'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock',  'TOTAL_COMPENSATION']
my_dataset_2 = data_dict

data2 = featureFormat(my_dataset_2, features_list2, sort_keys = True)

labels2, features2 = targetFeatureSplit(data2)

scale2 = preprocessing.MinMaxScaler()
features2 = scale2.fit_transform(features2)

comp = 15
pca2 = PCA(comp)
features2 = pca2.fit_transform(features2)


test_classifier(clf_DT, my_dataset_2, features_list2)


# In[23]:


#add to poi_id
clf = clf_DT
test_classifier(clf, my_dataset, features_list)


# In[24]:


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# In[ ]:




