import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

import regex as re
import time

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def model_scores(model, X_train, y_train, X_test, y_test): 
    ''' Evaluate the instantiated models for initial reviews
    '''
    model.fit(X_train, y_train)
    
    tr = model.score (X_train, y_train)
    te = model.score (X_test, y_test)
    cv = cross_val_score (model, X_train, y_train, cv=5).mean()
    cv2 = cross_val_score(model, X_test, y_test, cv=5).mean()

    y_pred = model.predict (X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    
    fit_status = lambda train_score, test_score: 'overfit' if train_score-test_score >0 else 'underfit' 
    bias_variance_status = lambda a, b: 'high variance' if a-b >0 else 'high bias' 
    
    time_start = time.time()
    time_end   = time.time()
    fit_time   = time_end - time_start

#     score_df = {
#         'model': model}

    score_df = {}

    score_df['model'] = model,
    score_df['accuracy score']= accuracy,
    score_df['cv train score'] = cv,
    score_df[ 'cv test score']= cv2,
    score_df['accuracy score']= accuracy,
    score_df['train score']=tr,
    score_df['test score']=te,
    score_df['train-test gap']=tr-te,
    score_df['model status']= fit_status(tr, te)
    score_df['bias vs variance'] = bias_variance_status(tr, cv)
    score_df['fit time'] = fit_time 
    
    return pd.DataFrame(score_df)


########################################################################
def gs_score (model, my_params, X_train_cv, y_train): 
    
    ''' GS Grid Search
    '''
    gs = GridSearchCV(model, param_grid=my_params, cv=5)
    gs.fit(X_train_cv, y_train)
    
    bp = gs.best_params_
    bs = gs.best_score_
    te = gs.score(X_test_cv, y_test)
    tr = gs.score(X_train_cv, y_train)
    
    gs_score_df = {}
    gs_score_df 
    
    gs_score_df['grid search model'] = model,
    #gs_score_df['best_params'] = bp
    gs_score_df['best_score'] = bs
    gs_score_df['train score']=tr,
    gs_score_df['test score']=te,
    gs_score_df['train-test gap']=tr-te

    print (gs.best_params_)
    return pd.DataFrame(gs_score_df)

########################################################################

def tokenizer_lemmatizer (df, text): 
    '''
    Initializing tokenizer and lemmatizer to handle NLP preprocessing. 
    1. breakdown the word by alphanumeric characters and dollar with number
    2. Create a list that appended with lemmatized posts and rejoin words by one string 
       alongside removing characters and numbers
    '''
    
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]')
    tokens = [tokenizer.tokenize(str(post).lower()) for post in (df[text])]
    
    
    lemmatizer = WordNetLemmatizer()
    lems = []
    for post in tokens:
        tok_post = []
        for word in post:
            tok_post.append(re.sub("[^a-zA-Z]", "", lemmatizer.lemmatize(word))) #Remove non-letter
        posts = " ".join(tok_post)
        lems.append(posts)
    
    words_not_used = ['wa', 've', 'ha', 'don']
    
    lems = [w for w in lems if not w in words_not_used] #stopwords.words('english')
    
    df[text] = lems #overwrite the df
    
    print (f'tokenizer processed: {len(tokens)}')
    print (f'lemmatizer processed: {len(lems)}')
    #return lemm

