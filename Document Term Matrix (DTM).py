#!/usr/bin/env python
# coding: utf-8

# # <center>HW 1: Document Term Matrix</center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# **Instructions**: 
# - Please read the problem description carefully
# - Make sure to complete all requirements (shown as bullets) . In general, it would be much easier if you complete the requirements in the order as shown in the problem description
# - Follow the Submission Instruction to submit your assignment

# **Problem Description**
# 
# In this assignment, you'll write a class and functions to analyze an article to find out the word distributions and key concepts. 
# 
# The packages you'll need for this assignment include numpy and pandas. Some useful functions: 
# - string: `split`,`strip`, `count`,`index`
# - numpy: `argsort`,`argmax`, `sum`, `where`
# - list: `zip`, `list`

# ## Q1. Define a function to analyze word counts in an input sentence 
# 
# 
# Define a function named `tokenize(text)` which does the following:
# * accepts a sentence (i.e., `text` parameter) as an input
# * splits the sentence into a list of tokens by **space** (including tab, and new line). 
#     - e.g., `it's a hello world!!!` will be split into tokens `["it's", "a","hello","world!!!"]`  
# * removes the **leading/trailing punctuations or spaces** of each token, if any
#     - e.g., `world!!! -> world`, while `it's` does not change
#     - hint, you can import module *string*, use `string.punctuation` to get a list of punctuations (say `puncts`), and then use function `strip(puncts)` to remove leading or trailing punctuations in each token
# * only keeps tokens with 2 or more characters, i.e. `len(token)>1` 
# * converts all tokens into lower case 
# * find the count of each unique token and save the counts as dictionary, i.e., `{world: 1, a: 1, ...}`
# * returns the dictionary 
#     

# In[1]:

import string
import pandas as pd
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:




def tokenize(text):
    
    vocab = {}
    #strip the space and split the sentence
    text_splited = (text.strip()).split()
    puncts = string.punctuation
    text_without_puncts = []
    # strip the punctuation for each word
    for i in text_splited:
        temp = i.strip(puncts)
    # only keeps tokens with 2 or more characters
    # lower the word's case and append them in array
        if len(temp) != 1:
            text_without_puncts.append(temp.lower())
    #count of each token 
            for i in text_without_puncts:
                vocab[i] = text_without_puncts.count(i) 

    return vocab


# In[3]:


# test your code
text = """it's a hello world!!!
           it is hello world again."""
tokenize(text)


# ## Q2. Generate a document term matrix (DTM) as a numpy array
# 
# 
# Define a function `get_dtm(sents)` as follows:
# - accepts a list of sentences, i.e., `sents`, as an input
# - uses `tokenize` function you defined in Q1 to get the count dictionary for each sentence
# - pools the words from all the strings togehter to get a list of  unique words, denoted as `unique_words`
# - creates a numpy array, say `dtm` with a shape (# of sentences x # of unique words), and set the initial values to 0.
# - fills cell `dtm[i,j]` with the count of the `j`th word in the `i`th sentence
# - returns `dtm` and `unique_words`

# In[4]:


def get_dtm(sents):
    
    dtm, all_words = None, None
    
    words = {}
    rows_of_text = len(sents) # rows of doc

    for idx in range(rows_of_text):
        words.update(tokenize(sents.loc[idx]))
    
    num_unique_words = len(words.keys()) # number of unique words
    all_words = sorted(words.keys()) # unique words

    dtm = np.ndarray(shape = (rows_of_text,num_unique_words), dtype= int)
    dtm.fill(0) 

    sentence = []
    for i in range(rows_of_text):
       sentence.append(sorted((tokenize(sents.loc[i])).items()))
       for j in range(num_unique_words):
            if j < len(sentence[i]):
                dtm[i, all_words.index(sentence[i][j][0])] = sentence[i][j][1]
        
    return dtm, all_words


# In[5]:


# A test document. This document can be found at https://hbr.org/2022/04/the-power-of-natural-language-processing
sents = pd.read_csv("sents.csv")
sents.head()


# In[6]:


dtm, all_words = get_dtm(sents.text)

# Check if the array is correct

# randomly check one sentence
idx = 3

# get the dictionary using the function in Q1
vocab = tokenize(sents["text"].loc[idx])
print(sorted(vocab.items(), key = lambda item: item[0]))

# get all non-zero entries in dtm[idx] and create a dictionary
# these two dictionaries should be the same
sents.loc[idx]
vocab1 ={all_words[j]: dtm[idx][j] for j in np.where(dtm[idx]>0)[0]}
print(sorted(vocab1.items(), key = lambda item: item[0]))


# ## Q3 Analyze DTM Array 
# 
# 
# **Don't use any loop in this task**. You should use array operations to take the advantage of high performance computing.

# Define a function named `analyze_dtm(dtm, words, sents)` which:
# * takes an array $dtm$ and $words$ as an input, where $dtm$ is the array you get in Q2 with a shape $(m \times n)$, $words$ contains an array of words corresponding to the columns of $dtm$, and $sents$ are the list of sentences you used in Q2.
# * calculates the sentence frequency for each word, say $j$, e.g. how many sentences contain word $j$. Save the result to array $df$ ($df$ has shape of $(n,)$ or $(1, n)$).
# * normalizes the word count per sentence: divides word count, i.e., $dtm_{i,j}$, by the total number of words in sentence $i$. Save the result as an array named $tf$ ($tf$ has shape of $(m,n)$).
# * for each $dtm_{i,j}$, calculates $tf\_idf_{i,j} = \frac{tf_{i, j}}{df_j}$, i.e., divide each normalized word count by the sentence frequency of the word. The reason is, if a word appears in most sentences, it does not have the discriminative power and often is called a `stop` word. The inverse of $df$ can downgrade the weight of such words. $tf\_idf$ has shape of $(m,n)$
# * prints out the following:
#     
#     - the total number of words in the document represented by $dtm$
#     - the most frequent top 10 words in this document    
#     - words with the top 10 largest $df$ values (show words and their $df$ values)
#     - the longest sentence (i.e., the one with the most words)
#     - top-10 words with the largest $tf\_idf$ values in the longest sentence (show words and values) 
# * returns the $tf\_idf$ array.
# 
# 
# 
# Note, for all the steps, **do not use any loop**. Just use array functions and broadcasting for high performance computation. To combine selected words and their values, you can use function `zip`.

# In[7]:


def analyze_dtm(dtm, words, sents):
    
    tfidf = None
    j = sum(dtm)
    top_10_index = np.argsort(j)[::-1][0:10] # index of the most frequent top 10 words
    top_10 = list(zip(words[top_10_index],j[top_10_index]))

    df = (dtm != 0).sum(axis = 0)
    top_10_index_df = np.argsort(j)[::-1][0:10] 
    top_10_df = list(zip(words[top_10_index_df],df[top_10_index_df]))
    
    len_sentence = dtm.sum(axis=1)
    longest_sentence = sents[np.argmax(len_sentence)]
    tf = (dtm.T / dtm.sum(axis=1)).T
    
    tfidf = tf/df
    
    long_s = tfidf[np.argmax(len_sentence)]
    top_10_index_tfidf = np.argsort(long_s)[::-1][0:10] # index of the most frequent top 10 words
    top_10_tfidf  = list(zip(words[top_10_index_tfidf ],long_s[top_10_index_tfidf ]))

   

    print("The total number of words:",sum(j))
    print("\nThe top 10 words:", top_10)
    print("\nThe top 10 words with highest df values:",top_10_df)
    print("\nThe longest sentence :\n",longest_sentence )
    print("\nThe top 10 words with highest tf-idf values in the longest sentece:",top_10_tfidf )
    # for the last question, I have 14 different words they all have same tf-idf, the final answer may vary with test answer SINCE mine was sorted alphabetically.
    
    return tfidf


# In[8]:


# convert the list to array so you can leverage array operations
words = np.array(all_words)

analyze_dtm(dtm, words, sents.text)


# 

# ## Q4. Find keywords of the document (Bonus) 
# 
# Can you leverage $dtm$ array you generated to find a few keywords that can be used to tag this document? e.g., AI, language models, tools, etc.
# 
# 
# Use a pdf file to describe your ideas and also implement your ideas.

# ## Put everything together and test using main block

# In[9]:


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    # Test Question 1
    text = """it's a hello world!!!
           it is hello world again."""
    print("Test Question 1")
    print(tokenize(text))
    
    
    # Test Question 2
    print("\nTest Question 2")
    sents = pd.read_csv("sents.csv")
    
    dtm, all_words = get_dtm(sents.text)
    print(dtm.shape)
    
    
    #3 Test Question 3
    print("\nTest Question 3")
    words = np.array(all_words)

    tfidf= analyze_dtm(dtm, words, sents.text)
    


# In[ ]:





# In[ ]:




