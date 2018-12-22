# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 20:52:56 2018

@author: Dell
"""

from sklearn.base import ClassifierMixin,TransformerMixin,BaseEstimator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,manhattan_distances
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import glob



class customKNN(ClassifierMixin):
    
    def __init__(self,label_to_vect_dict=dict(),sim='cs'):
        sim_fn={'cs':cosine_similarity,
                'ed':euclidean_distances,
                'md':manhattan_distances
                }
        self.similarity_fn=sim_fn[sim]

        self.label_to_vect_dict=label_to_vect_dict
        
        
    def fit(self,X,y):
        """
        computes centroid and stores in dictionary 
        
        X-np.array-->Text embedings ,vectors etc
        Y-labels ,list of numbers -encoded
        
        """
        print('inside customknn')
        print(X,y)
        
        for i in set(y):
            if i==np.nan:
                raise 'label contains Nan'
            ith_class_X=X[y==i]
#            np.sum(ith_class_X)
            self.label_to_vect_dict[str(i)]=np.sum(ith_class_X,axis=0)/ith_class_X.shape[0]
        return self
    
    
    def predict(self,X):
        """
        Using computed dictionary predicts the nearest match to the
        given label
        
        X-np.array-->Text embedings,vectors etc
        
        sim functions
        'cs':cosine_similarity,
        'ed':euclidean_distances,
        'md':manhattan_distances
        """
        
        
        ret=[]
        self.prob_array=np.zeros((X.shape[0],len(self.label_to_vect_dict)))
        
        for ins in X:
#            max_prob=0
            max_similarity=-999
            similar_label=0
#            similarity_dict=dict()
            for label,centroid in self.label_to_vect_dict.items():
                sim_array=self.similarity_fn(centroid.reshape(1,-1),ins.reshape(1,-1))
                sim_array=np.absolute(sim_array)
                sim_score=sim_array.sum()/sim_array.shape[0]
#                print(sim_score)
#                similarity_list[label]=sim_score
#                print(sim_score)
                
                if max_similarity < sim_score:
                    max_similarity=sim_score
                    similar_label=int(label)
              
#                
            ret.append(similar_label)

#                   
                    
        return np.array(ret)
    
    def fit_with_new_label(self,X,y):
        """
        To fit already fitted model 
        
        X-np.array-->Text embedings ,vectors etc
        Y-labels ,list of numbers -encoded
        
        
        """
        
        for i in set(y):
           
            ith_class_X=X[y==i]
#            np.sum(ith_class_X)
            self.label_to_vect_dict[str(i)]=np.sum(ith_class_X,axis=0)/ith_class_X.shape[0]
        return self
    
    
    def get_centroid(self):
        """
        Outputs computed centroid by fit method
        """
        
        return self.label_to_vect_dict.copy()
    
    def predict_proba(self,X):
        print('x',X)
        prob_array=np.zeros((X.shape[0],len(self.label_to_vect_dict)))
        for ins_idx,ins in enumerate(X):
            
            for label,centroid in self.label_to_vect_dict.items():
                sim_array=self.similarity_fn(centroid.reshape(1,-1),ins.reshape(1,-1))
                sim_score=sim_array.sum()/sim_array.shape[0]
                
                prob_array[ins_idx,len(self.label_to_vect_dict)]=sim_score.reshape(1,-1)
        
        return prob_array.copy()
    

class ParagraphVectors(BaseEstimator,TransformerMixin):
    
    
    def __init__ (self,max_epochs=100,
                  vec_size=100,
                  alpha=0.025,
                  dm=1,
                  filename='user_story',
                  **doc2vec_args):
        
        self.max_epochs=max_epochs
        self.vec_size=vec_size
        self.alpha=alpha
        self.dm=dm
        self.filename=filename
        self.DIRECTORY_PATH=r'tmp\\'

        
    def fit_transform(self,X,y=None):
        self.fit(X)
        return self.transform(X)
    
    def get_tagged_data(self,text_series):


        data=text_series.tolist()
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
        return tagged_data

        
       
    def fit(self,text_series,**doc2vec_args):
        
        """
        Distribuited memory vectors
        """
        
        if not glob.glob(self.DIRECTORY_PATH+self.filename+'*.model'):
            
        
            tagged_data=self.get_tagged_data(text_series)
            model = Doc2Vec(size=self.vec_size,
                            alpha=self.alpha, 
                            min_alpha=0.025,
                            min_count=1,
                            dm =self.dm,
                            **doc2vec_args)
              
            model.build_vocab(tagged_data)
        else:
            
            tagged_data=self.get_tagged_data(text_series)

            model= Doc2Vec.load("{}{}_d2v.model".format(self.DIRECTORY_PATH,self.filename))

            
        for epoch in range(self.max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.iter)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha
        
        model.save("{}{}_d2v.model".format(self.DIRECTORY_PATH,self.filename))
        print("Model Saved to {}{}".format(self.DIRECTORY_PATH,self.filename))
        
   
    
    def transform(self,test_sent_list):
       
        if isinstance(test_sent_list,pd.core.series.Series):
            test_sent_list=test_sent_list.tolist()

            
        if not (isinstance(test_sent_list,list) or isinstance(test_sent_list,np.ndarray)):            
            test_sent_list=[test_sent_list]
        
        
        model= Doc2Vec.load("{}{}_d2v.model".format(self.DIRECTORY_PATH,self.filename))
        pred=[]
        for sent in test_sent_list:
            pred.append(model.infer_vector(sent))
        return np.vstack([pred])

#import pickle
#pipe=Pipeline(steps=[('pv',ParagraphVectors()),('knn',customKNN())])
####
#pipe.fit(user_story['Summary'],pd.factorize(user_story['Priority'])[0])
###dictionary=pipe.named_steps.knn.get_centroid()
###pickle.dump(dictionary,open(r'dict_knn_centroid.pkl','wb'))
#
#val=np.array(pipe.predict(user_story['Summary']))
##
#np.mean(y==val)
#pv=ParagraphVectors()
#pv.fit(user)

#from sklearn.pipline import Pipline

















    