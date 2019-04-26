# Project-4-Disaster-Classification : Leveraging News and Media for Situational Awareness

### Contributors: 
* Bill Yu
* Manu Kalia
* Evelyn Li
* Kun Guo

---
## Table of Contents 

This Notebook is broken down into different sections for analysis purpose. The following links are connected to differenct section within the Notebook for simple navigation. 

---

## Notebooks:
- [0. Custom_stop_words](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/0.%20Custom_stop_words.ipynb)
- [1. NewAPI_exploration_Finalized](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/1.%20NewAPI_exploration_Finalized%20.ipynb)
- [2.a EDA Cleaning and Baseline Model](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/2.a%20EDA%20Cleaning%20and%20Baseline%20Model.ipynb)
- [2.b EDA Cleaning and Baseline Model](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/2.b%20EDA%20%26%20Visualization%20.ipynb)
- [2.c Doc2vevc_Model_Tryout](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/2.c%20Doc2vevc_Model_Tryout.ipynb)
- [2.d LDA Tryout](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/2.d%20LDA%20Tryout.ipynb)
- [3a. CountVectorizer + TFIDF for EDA modeling](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/3a.%20CountVectorizer%20%2B%20TFIDF%20for%20EDA%20modeling%20.ipynb)
- [3b. TFIDF Final Model and Tunings](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/3b.%20TFIDF%20Final%20Model%20and%20Tunings.ipynb)
- [4. Model_Testing](https://git.generalassemb.ly/billyu/Project-4-Disaster-Classification/blob/master/code/4.%20Model_Testing%20.ipynb)

---

## Problem Statement: 

---

During a major disaster, it is essential to provide relevant information regarding the disaster to the public and responder to gain situational awareness during the event. However, during major disasters, news comes from tens to hundreds of news sources and channels since each news organization will have their cover of the disaster from a different angle. Although the information is useful, this also makes it more difficult for the public and responders to captures all valuable and relevant information regarding the disaster since they have to go to different sources to gather information. Recognizing there is an inefficiency within the overwhelmingly rich dataset in the world, we decided to build a tool that allows the website editor to set these keywords and search queries based on the keywords the model provided. After collecting the news, editors can run the content through the trained model where the model will classify whether or not the news content is relevant to the disaster. 

---


## Datasets: 

- 11000 news collected from `News.api`

|  Column Name  |                          Description                          |
|:-------------:|:-------------------------------------------------------------:|
|     author    |                  author of the news articles                  |
|    content    |                      content of the news                      |
|  description  |                 brief description of the news                 |
|  publishedAt  |            timestamp of when the news is published            |
|     source    |                      source of the news                       |
|   source_id   |                     source_id of the news                     |
|  source_name  |                    source link of the news                    |
|     title     |                       title of the news                       |
|     types     |                       type of disasters                       |
|      url      |                      url to the news page                     |
|   urlToImage  |                       images in the url                       |
| yes_disaster  | classification of disaster 1 = yes_disaster  0 = no_disaster  |

---



## Executive Model Summary 
---


Classification estimators: 

9 different models deployed, 3 vectorization methods, 2 sets of features (Content and Content+Description)
 
Hyperparameter Tuning:

1. Stop words and customized stop words (remove locations names, like countries, regions, states, cities)
2. Tokenize and Lemmatize
3. Features (content only, and content + description)
4. Grid Search
 
Final model selection: 

1. TFIDF + custom location stop words + Logistic Regression
2. Pull out key parameters (top feature words & coefficients)
3. Compare the coefficients and feature name before and after the customized stop words


---

## Conclusion & Recommendation 
---

1.Rank-ordered feature words:  final model creates a set of important words and their associated coefficients, the client can apply the top words to classify all ingested news articles, and display "disaster=yes" items on a webpage.
Keywords to use:  top coefficient words from the final logistic regression model

Set up a scheduled script (as in CRON) to ingest all news posts (and/or tweets) 1-4 times per day.  Use this downloaded set of articles as a new 'test' dataset for our trained and fitted model.

Display all the news items that are returned as positive for the disaster class.  There will be some false positives that are posts containing disaster words but are not necessarily about a disaster in progress.
 
2. Rerun the model & update the key features list.  Periodically (2-3 times per year) use the top 50 or so key features to download more articles for training the classifier(s). The update cycle will collect news articles, and check for duplicates to articles in the existing training dataset. Re-run the Naive-Bayes and Logistic Regression models to keep the keywords set updated.

 
3. Continuously grow the classified training dataset.  Collect large new sets of articles and characterize each one as "disaster" or not-disaster" by human inspection.  This is a time and labor-intensive task, so it is important to minimize the labor expense whenever possible...
a. Local contests
b. Internships
c. Disaster assistance volunteer days

4. Continuously grow the dataset also by flagging all news articles on known disaster days, and store for future incorporation.  The articles on the day of a disaster (and perhaps for a few days after) will have a much higher number of 'disaster=yes' articles, which are very useful for the training set.
 
5. Future model enhancement
a. Secondarily classify the disaster types (wildfires, storms, floods, etc.)
b. Identify a "Disaster Condition" using the number of articles classified as 'yes_disaster' during a certain time-period (perhaps one day or several hours)


## References 
---

- Shperber, Gidi. (2017, Jul 25). "A gentle introduction to Doc2Vec". [https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e]
- Li, Susan. (2018, Sep 17). "Multi-Class Text Classification with Doc2Vec & Logistic Regression". [https://towardsdatascience.com/multi-class-text-classification-with-doc2vec-logistic-regression-9da9947b43f4]