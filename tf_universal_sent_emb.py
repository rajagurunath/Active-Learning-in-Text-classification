# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 19:46:30 2018

@author: gurunath.lv
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd

def universal_sent_embeddings(text_series):
    """
    
    Taking too much time to process but produces good
    quality word embeddings using universal sentance embeddings
    """
    import tensorflow as tf
    import tensorflow_hub as hub

    
    test_decriptions_list=text_series.tolist()
    
    embed = hub.Module(r'C:\Users\gurunath.lv\AppData\Local\Temp\tfhub_modules\c6f5954ffa065cdb2f2e604e740e8838bf21a2d3')
    tf.logging.set_verbosity(tf.logging.ERROR)
    
    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      test_descriptions_embeddings = session.run(embed(test_decriptions_list))
    return test_descriptions_embeddings

def transform_using_tfidf(text_series):
    tfidf=TfidfVectorizer(stop_words='english')
    array=tfidf.fit_transform(text_series.tolist()).toarray()
    return array,tfidf
    


def similarity_measure(inp_sent,array,tfidf,top_n):
    inp_vec=tfidf.transform([inp_sent]).toarray()
    
    cs=cosine_similarity(inp_vec,array)
    top_match_index=np.flip(np.argsort(cs,axis=1)[:,-top_n:],axis=1)
    return top_match_index
    


def get_similar_records(inp_sent,total_text,top_n=10):
    array,tfidf=transform_using_tfidf(total_text)
    top_match_index=similarity_measure(inp_sent,array,tfidf,top_n)
    return total_text.iloc[top_match_index.ravel()]
    

if __name__=='__main__':
    user_story=pd.read_csv(r'D:\Testing_frameworks\Testcase-Vmops\Insight\data\interim\filtered_user_story_by_priority.csv',encoding='ISO-8859-1')
    get_similar_records(user_story['Summary'][2],user_story['Summary'])
    
    
    
    
    
    
    
    
    
    