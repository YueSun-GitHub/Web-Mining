#!/usr/bin/env python
# coding: utf-8

# # <Center> HW 4: Classification </center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# In this assignment, we use classification to identify deceptive comments. This assignment needs the following two data files:
# - hw4_train.csv: dataset for training
# - hw4_test.csv: dataset for testing
#     
# Both of them have samples in the following format. The `text` column contains documents and the `label` column gives the sentiment of each document.
# 
# |label | text |
# |------|------|
# |1|  when i first checked the hotel's website and r...|
# |1|  I had really high hopes for this hotel. The lo...|
# |0|  My experiences at the Fairmont Chicago were le...|
# |...|...|
# 

# In[3]:


import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import numpy as np
np.set_printoptions(precision=3)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc,precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


# In[4]:


train = pd.read_csv("hw4_train.csv")
train.head(3)
test = pd.read_csv("hw4_test.csv")
test.head(3)


# ## Q1 Text Vectorization and Classification 
# 
# For classification, the first step is to compute the word TF-IDF weights for each document. A few options can be configured as given below.
# 
# 
# Define a function `classify(train_docs, train_y, test_docs, test_y, classifier = 'naive bayes', binary=False, ngrams = (1,1), stop_words='english', min_df=1, show_plots=False)`, where
# 
# - `train_docs`: is a list of documents for training.
# - `train_y`: is the ground-truth labels of training documents.
# - `test_docs`: is a list of documents for test.
# - `test_y`: is the ground-truth labels of test documents.
# - `classifier`: the name of classification algorithm. Two possible values: 'svm','naive bayes'. The default value is 'naive bayes'.
# - `binary`: if true, within a document, the term frequency of a word is binarized to 1 if present and 0 otherwise. If False, the regular term frequency is considered. The default is False.
# - `ngrams`: an option to include unigrams, bigrams, ..., nth grams. The default is (1,1), i.e., only unigrams used.
# - `stop_words`: indicate whether stop words should be removed. The default value is 'english', i.e. remove English stopwords.
# - `min_df`: only tokens with document frequency above this threshold can be included. The default is 1.
# - `show_plots`: controls whether to show classification report AND plots. The default is False.
# 
# 
# This function does the following:
# - Fit a `TfidfVectorizer` using `train_docs` with options `stop_words, min_df, ngrams, binary` as specified in the function inputs. Extract features from `train_docs` using the fitted `TfidfVectorizer`.
# - Train a classifier by the specified `classifier` algorithm using the extracted features from `train_docs` and labels from `train_y`.
# - Transform `test_docs` by the fitted `TfidfVectorizer` (hint: use function `transform` not `fit_transform`).
# - Predict the labels for `test_docs` with trained model.
# - If `show_plots` is True,
#     - Print the classification report.
#     - Plot the AUC score and PRC score (or Average Precision) for class 1 on the test dataset. On the plot, specify xlabel, ylabel on axis, and the scoring metrics (AUC/PRC/Average Precision) on the title.
#     - Note, if the classifier is 'svm', please use `decision_function` instead of predict_prob function. The details can be found at https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.decision_function.
# - Return the `TfidfVectorizer` and the trained model.
#     
# 
# Test your function with following cases:
# - `stop_words = 'english', binary = False, classifier='naive bayes', show_plots = True`
# - `stop_words = 'english', binary = False, classifier='svm', show_plots = True`

# In[5]:


def classify(train_docs, train_y, test_docs, test_y,                 classifier = 'naive bayes',
                binary=False, ngrams = (1,1), \
                stop_words='english', min_df=1, \
                show_plots=True):

    clf, tfidf_vect = None, None
    
    #Tfidfvectorizer
    tfidf_vect = TfidfVectorizer(stop_words = stop_words, min_df = min_df, binary = binary, ngram_range = ngrams )  
    #Transform
    train_doc_vector = tfidf_vect.fit_transform(train_docs)
    test_doc_vector = tfidf_vect.transform(test_docs)
    
    if classifier == 'naive bayes':
        clf = MultinomialNB().fit(train_doc_vector, train_y)
        predict_p = clf.predict_proba(test_doc_vector)
        y_pred = predict_p[:,1]
        
    if classifier == 'svm':
        clf = svm.LinearSVC().fit(train_doc_vector, train_y)
        predict_p = clf.decision_function(test_doc_vector)
        y_pred = predict_p
        
    binary_y = np.where(test_y == 1,1,0)
    predicted = clf.predict(test_doc_vector)  
    print(classification_report(test_y, predicted))

    if show_plots:
        fpr, tpr, thresholds = roc_curve(binary_y, y_pred)
        precision, recall, thresholds = precision_recall_curve(binary_y,y_pred)

        f, ax = plt.subplots(1,2, figsize = (15,5))

        ax[0].set_title('auc')
        ax[0].set(xlabel = 'False Positive Rate',ylabel = 'True Positive Rate')
        ax[0].plot(fpr, tpr, color='darkorange', lw=2)
        ax[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax[0].set(xlim = [-0.05, 1.05], ylim = [-0.05, 1.05])

        ax[1].set_title('prc'); 
        ax[1].set(xlabel = 'Recall', ylabel = 'Precision')
        ax[1].plot(recall, precision, color='darkorange', lw=2)
        ax[1].set(xlim = [-0.05, 1.05], ylim = [-0.05, 1.05])

        print("AUC: {:.2%}".format(auc(fpr, tpr)) , ", PRC: {:.2%}".format(auc(recall, precision)))

    return  clf, tfidf_vect


# In[6]:


clf, vectorizer = classify(train["text"], train["label"],
                           test["text"], test["label"],
                           stop_words = 'english', binary = False, 
                           classifier='naive bayes', show_plots = True)


# In[7]:


clf, vectorizer = classify(train["text"], train["label"],
                           test["text"], test["label"],
                           stop_words = 'english', binary = False, 
                           classifier='svm', show_plots = True)


# ## Q2: Search for best parameters
# 
# From Q1, you may find there are many possible ways to configure parameters. Next, let's use grid search to find the optimal parameters.
# 
# - Define a function `search_para(docs, y, classifier = 'naive bayes')` where `docs` are training documents, `y` is the ground-truth labels, and `classifier` is the model you use.
# - This function does the following:
#     - Create a pipleline which integrates `TfidfVectorizer` and the classifier as specified by parameter `classifier` . 
#     - Define the parameter ranges as follow: 
#         - `stop_words: [None, 'english']`
#         - `min_df:[1, 3]`
#         - `ngram_range:[(1,1), (1,2), (1,3)]`
#         - `binary: [True, False]`
#     - Set the scoring metric to `f1_macro`. 
#     - Use `GridSearchCV` with `4-fold cross validation` to find the best parameter values based on the training dataset. 
#     - Print the values of the `best` parameters combination. 
#     
# - Call this function to find `the best parameters combination` for linear SVM and Naive Bayes models. 
# - Call the function `classify` again to use `the best parameters combination`
# 
# 
# Please briefly answer the following: 
# - Compared with the model in Q1, how is the performance improved on the test dataset?
# - Why do you think the new parameter values help deceptive comment classification?

# In[20]:


def search_para(docs, y, classifier = 'naive bayes'):
       
    if classifier == 'svm':

        text_clf = Pipeline([('tfidf', TfidfVectorizer()),('clf', svm.LinearSVC())])

    if classifier == 'naive bayes':

        text_clf = Pipeline([('tfidf', TfidfVectorizer()), 
                             ('clf', MultinomialNB())])

    parameters = {'tfidf__min_df':[1,3],
                  'tfidf__stop_words':[None,"english"],
                  'tfidf__ngram_range':[(1,1), (1,2), (1,3)],
                  'tfidf__binary': [True, False]}

    metric =  "f1_macro"

    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=4)
    gs_clf = gs_clf.fit(docs, y)

    for param_name in gs_clf.best_params_:
        
        print("{0}:\t{1}".format(param_name,                                     gs_clf.best_params_[param_name]))

    print("best f1 score: {:.3f}".format(gs_clf.best_score_)) 


# In[21]:


search_para(train["text"], train["label"])


# In[22]:


search_para(train["text"], train["label"], classifier ='svm')


# In[133]:


clf, vectorizer = classify(train["text"], train["label"],
                              test["text"], test["label"],
                              stop_words= None, min_df = 3, binary=True,
                              ngrams = (1,2), classifier = 'naive bayes', show_plots=True)


# In[134]:


clf, vectorizer = classify(train["text"], train["label"],
                              test["text"], test["label"],
                              stop_words= 'english', min_df = 2, binary=True,
                              ngrams = (1,3), classifier = 'svm', show_plots=True)


# In Q1, we get 86% and 87% in Naive Bayes and SVM classification. After we tune the parameters, we increased the value to 89% and 88%.
# I think the reason new parameters help deceptive comment it because some deceptive are bigrams term and we haven’t count them in Q1. Also, we kept the stop words because there are some negation words we should kept them in our model. If we remove the stop words it will lose some information.
# 

# ## Q3. Impact of Sample Size 
# 
# 
# This task is to help you understand the impact of sample size on classifier performance. 
# 
# Define a function `show_sample_size_impact(train_docs, train_y)` where:
# - `train_docs`: is a list of documents for training.
# - `train_y`: is the ground-truth labels of training documents.
#     
# Conduct the experiment as follows:    
# - Starting with 100 samples, in each round you build a classifier with 100 more samples. i.e. in round 1, you use samples from 0 to 100, and in round 2, you use samples from 0 to 200, …, until you use up all samples. 
# - In each round, you'll conduct `4-fold cross validation` for both Naive Bayes and Linear SVM algorithms with all `default paramater values` in the TFIDF vectorizer and classifiers. Record the average testing F1-macro score.
# - Plot a line chart to show the relationship between sample size and the F1-macro score for SVM and Naive Bayes models. 
# - This function has no return.
#     
#     
# - Write your analysis on the following:
#     - How does sample size affect each classifier’s performance? 
#     - If it is expensive to collect and label samples, can you decide an optimal sample size with model performance and the cost of samples both considered? 
#   

# In[69]:


def show_sample_size_impact(train_docs, train_labels):
    

    clf_NB = Pipeline([('tfidf', TfidfVectorizer()), 
                                 ('clf', MultinomialNB())])    
    clf_svm = Pipeline([('tfidf', TfidfVectorizer()), 
                                 ('clf', svm.LinearSVC())])
    metric = "f1_macro"

    parameters = {} #default paramater values

    gs_clf_NB= GridSearchCV(clf_NB,param_grid = parameters,scoring=metric, cv = 4)
    gs_clf_svm= GridSearchCV(clf_svm,param_grid = parameters,scoring=metric, cv = 4)

    seq = list([100,200,400,500,600,700,800,900,1020])

    list_NB,list_svm = [],[]

    for i in seq:

        temp_docs = train_docs[0:i]
        temp_labels = train_labels[0:i]

        gs_clf_NB = gs_clf_NB.fit(temp_docs, temp_labels)
        gs_clf_svm = gs_clf_svm.fit(temp_docs, temp_labels)

        list_NB.append(gs_clf_NB.best_score_)
        list_svm.append(gs_clf_svm.best_score_)

    plt.plot(seq, list_NB, color='lightblue', lw=2, label = "naive bayes")
    plt.plot(seq, list_svm, color='darkorange', lw=2, label ='svm')
    plt.xlabel("size")
    plt.legend(loc = 4)  


# In[70]:


show_sample_size_impact(train["text"], train["label"])


# The model with larger sample size has better performance.
# 
# I would choose sample size 500 as the optimal sample size. Since it only cost half of total expense and it reached to 85% f1-macro score. 
# 

# ## Q4 (Bonus): Model Interpretation 
# 
# For this dataset, both Naive Bayes and SVM model can provide good classification accuracy. The question is, how the models conclude that a document is deceptive. What features have the discriminative power? Do your research to find the most descriminative features in a document. To illustrate, you can randomly select a few samples from the test subset to highlight these most discriminative featues.

# In[83]:


test['text'][479] # label 1 
test['label'][479]

test['text'][227] # label 1
test['label'][227]

test['text'][239] # label 0
test['label'][239]


# In[ ]:




