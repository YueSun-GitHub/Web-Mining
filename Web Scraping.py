#!/usr/bin/env python
# coding: utf-8

# # <center>HW2: Web Scraping</center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work or let someone copy your solution (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. No last minute extension of due date. Be sure to start working on it ASAP! </div>

# ## Q1. Collecting Movie Reviews
# 
# Write a function `getReviews(url)` to scrape all **reviews on the first page**, including, 
# - **title** (see (1) in Figure)
# - **reviewer's name** (see (2) in Figure)
# - **date** (see (3) in Figure)
# - **rating** (see (4) in Figure)
# - **review content** (see (5) in Figure. For each review text, need to get the **complete text**.)
# - **helpful** (see (6) in Figure). 
# 
# 
# Requirements:
# - `Function Input`: book page URL
# - `Function Output`: save all reviews as a DataFrame of columns (`title, reviewer, rating, date, review, helpful`). For the given URL, you can get 24 reviews.
# - If a field, e.g. rating, is missing, use `None` to indicate it. 
# 
#     
# ![alt text](IMDB.png "IMDB")
# 

# In[65]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import requests
import json
import pandas as pd
from bs4 import BeautifulSoup     


# In[68]:


def getReviews(page_url):
    
    reviews = None

    page = requests.get(page_url)   
    rows = []

    if page.status_code==200:        
        soup = BeautifulSoup(page.content,'html.parser')

        divs = soup.select("div.lister-item-content")

        for idx, div in enumerate(divs):
            
            title: None
            user: None
            date: None
            rating: None
            content: None
            vote: None 

            p_title = div.select("a.title")
            if p_title!=[]:
                title = p_title[0].get_text()
                if len(title) != 0:
                    title = title
                else:
                    title = "None"
            
            
            p_user = div.select("span.display-name-link")
            if p_user!=[]:
                user = p_user[0].get_text()
                if len(user) != 0:
                    user = user
                else:
                    user = "None"
            

            p_date = div.select("span.review-date")
            if p_date!=[]:
                date = p_date[0].get_text()
                if len(date) != 0:
                    date = date
                else:
                    date = "None"
           

            p_rating = div.select("div.ipl-ratings-bar > span > span:nth-child(2)")
            if p_rating!=[]:
                rating = p_rating[0].get_text()
                if len(rating) != 0:
                    rating = rating
                else:
                    rating = "None"
            

            p_content = div.select("div.content div.show-more__control")
            if p_content!=[]:
                content = p_content[0].get_text()
                if len(content) != 0:
                    content = content
                else:
                    content = "None"
            

            p_vote = div.select("div.actions.text-muted")
            if p_vote!= []:
                vote = p_vote[0].get_text().split("\n")[1]
                if len(vote) != 0:
                    vote = vote
                else:
                    vote = "None"
            
        
            rows.append((title,user,date,rating,content,vote))
            
    cols = ["title","user","date","rating","review content","helpful"]
    reviews = pd.DataFrame(rows, columns= cols)        

    return reviews


# In[71]:


# Test your function
page_url = 'https://www.imdb.com/title/tt1745960/reviews?sort=totalVotes&dir=desc&ratingFilter=0'
reviews = getReviews(page_url)

print(len(reviews))
reviews.head()


# ## Q2 (Bonus) Collect Dynamic Content
# 
# Write a function `get_N_review(url, webdriver)` to scrape **at least 100 reviews** by clicking "Load More" button 5 times through Selenium WebDrive, 
# 
# 
# Requirements:
# - `Function Input`: book page `url` and a Selenium `webdriver`
# - `Function Output`: save all reviews as a DataFrame of columns (`title, reviewer, rating, date, review, helpful`). For the given URL, you can get 24 reviews.
# - If a field, e.g. rating, is missing, use `None` to indicate it. 
# 
# 

# In[123]:


def getReviews(page_url, driver):
    
    reviews = None
    driver.get(page_url)
    # click load more 5 times

    fivetimes_click = 5
    click_num = 0

    while click_num < fivetimes_click:
        
        try:
            WebDriverWait(driver,20).until(expected_conditions.element_to_be_clickable((By.ID,"load-more-trigger")))
            button = driver.find_element(By.ID,"load-more-trigger")
            button.click()
            click_num += 1

        except: 
            print("Unable to load more page")
            break
    divs = driver.find_elements(By.CSS_SELECTOR,"div.lister-item-content")
    rows = []
    for idx, div in enumerate(divs):

        p_title = div.find_element(By.CSS_SELECTOR,"a.title")
        if p_title != []:
            title = p_title.text
            if len(title) != 0:
                title = title
            else:
                title = "None"

        p_user = div.find_element(By.CSS_SELECTOR,"span.display-name-link")
        if p_user!=[]:
            user = p_user.text
            if len(user) != 0:
                user = user
            else:
                user = "None"
                

        p_date = div.find_element(By.CSS_SELECTOR,"span.review-date")
        if p_date!=[]:
            date = p_date.text
            if len(date) != 0:
                date = date
            else:
                date = "None"
            

        p_rating = div.find_element(By.CSS_SELECTOR,"div.ipl-ratings-bar > span > span:nth-child(2)")
        if p_rating!=[]:
            rating = p_rating.text
            if len(rating) != 0:
                rating = rating
            else:
                rating = "None"
                
        p_content = div.find_element(By.CSS_SELECTOR,"div.content div.show-more__control")
        if p_content!=[]:
            content = p_content.text
            if len(content) != 0:
                content = content
            else:
                content = "None"

        p_vote = div.find_elements(By.CSS_SELECTOR,"div.actions.text-muted")
        for i in p_vote:
            vote = i.text.split("Was")[0]
            if len(vote) != 0:
                vote = vote
            else:
                vote = "None"

        
        rows.append((title,user,date,rating,content,vote))
                
    cols = ["title","user","date","rating","review content","helpful"]
    reviews = pd.DataFrame(rows, columns= cols)        

    
    return reviews


# In[124]:


# Test the function

executable_path = 'driver/chromedriver'

driver = webdriver.Chrome(executable_path=executable_path)

page_url = 'https://www.imdb.com/title/tt1745960/reviews?sort=totalVotes&dir=desc&ratingFilter=0'
reviews = getReviews(page_url, driver)

driver.quit()

print(len(reviews))
reviews.head()

