# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 18:49:13 2018

@author: gurunath.lv
"""

#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
#from __future__ import unicode_literals, print_function
#import plac
import random
from pathlib import Path
#import thinc.extra.datasets

import spacy
from spacy.util import minibatch, compounding

import pandas as pd
import os
from lime.lime_text import LimeTextExplainer
import numpy as np
import glob
from custom_classifier import customKNN


DIRECTORY_PATH=r'tmp\\'

def return_text_categorizer(filename,model=None):
    
    path=glob.glob(DIRECTORY_PATH+filename)
    if len(path)>=10:
        nlp = spacy.load(path[0])  # load existing spaCy model
        
        
#    if model is not None:
#        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % path)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    return nlp

def load_data_for_train(already_trained_label,train_data):
    
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
#    train_data, _ = thinc.extra.datasets.imdb()
#    train_data=user_story
#    random.shuffle(train_data)
#    train_data = train_data[-limit:]
    default_dict=dict()

    texts, labels =train_data.iloc[:,0].values,train_data.iloc[:,1]
    gt_labels = labels.tolist()
    categories=list(set(gt_labels))
    
    if len(already_trained_label)>0:
        categories.extend(already_trained_label)
        categories=list(set(categories))
    
    
    
    for k in categories:
        default_dict[k]=False
    
#    default_dict.fromkeys(categories)
#    cats_list=prepare_cat_data(gt_labels)
    
    cats_list=[]
#    gt_labels=[new_label]*len(train_data)
    for cat in gt_labels:
        
        tmp_dict=default_dict.copy()
        tmp_dict[cat]=True
        cats_list.append(tmp_dict)
    
#    split = int(len(user_story) * split)
#    cats = [{str(y):True} for y in cats]
    return texts,cats_list

def train_categorizer(nlp,train_data,file_name,n_iter=20):
    categories=set(train_data[train_data.columns[1]])
    
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
        # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')
        print('else part')
        
    print(textcat.labels)
    print(categories)
   
    already_trained_label=textcat.cfg['labels']
    config_dict=textcat.cfg
    for cat in categories:
        config_dict['labels'].append(str(cat))
    for cat in categories:
        print(cat)
        textcat.add_label(cat)
    textcat.cfg=config_dict
    print(textcat.labels)
    texts,cats_list=load_data_for_train(already_trained_label,train_data)
    train_data = list(zip(texts,
                          [{'cats': cats} for cats in cats_list]))
    
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()
#    optimizer = textcat.begin_training()
        print("Training the model...")
        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
#            print(batches)
            for batch in batches:
#                print(batch)
                texts, annotations = zip(*batch)
#                print(annotations)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, texts, cats_list)
            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))
    
    nlp.to_disk(os.path.join(r'{}'.format(DIRECTORY_PATH),file_name))
    
    print('CNN trained and saved to')
    print(os.path.join(r'{}'.format(DIRECTORY_PATH),file_name))


def predict(sent_list,file_name):
#    Path(output_dir)
    global fileName
    nlp=spacy.load(Path(os.path.join(r'{}'.format(DIRECTORY_PATH),file_name+'\\')))
    fileName=file_name
    pred=[]
    for sent in sent_list:
        doc=nlp(sent)
        explain_prediction(sent,file_name)
        pred.append(doc.cats)
    return pred
    
#model=spacy.load(r'{}filtered_user_story_by_priority'.format)

def get_categories(sent,file_name):
    nlp=spacy.load(Path(os.path.join(r'{}'.format(DIRECTORY_PATH),file_name+'\\')))
    doc=nlp(sent)
    return list(doc.cats.keys())

def get_file_name():
    return fileName

def spacy_prediction(sent_list):
    file_name=get_file_name()
    nlp=spacy.load(Path(os.path.join(r'{}'.format(DIRECTORY_PATH),file_name+'\\')))
    
    ret=[]
    for sent in sent_list:
        doc=nlp(sent)
        ret.append(list(doc.cats.values()))
    
    return np.vstack(ret)
#    return spacy_pred
        

def explain_prediction(sent,file_name):
#    vect=transform_inp_sent_to_vect(sent)
    labels=get_categories(sent,file_name)
    explainer = LimeTextExplainer(class_names=labels)
    
    exp = explainer.explain_instance(sent, spacy_prediction,labels=[0,1])
    return exp.save_to_file(r'{}explanation.html'.format(DIRECTORY_PATH))


def train_knn(X,y):
    pass
    



#def main(model=None, output_dir=r'D:\Testing_frameworks\Testcase-Vmops\Insight\models\spacy_models\\', n_iter=20, n_texts=2000):
#    if model is not None:
#        nlp = spacy.load(model)  # load existing spaCy model
#        print("Loaded model '%s'" % model)
#    else:
#        nlp = spacy.blank('en')  # create blank Language class
#        print("Created blank 'en' model")
#
#    # add the text classifier to the pipeline if it doesn't exist
#    # nlp.create_pipe works for built-ins that are registered with spaCy
#    if 'textcat' not in nlp.pipe_names:
#        textcat = nlp.create_pipe('textcat')
#        nlp.add_pipe(textcat, last=True)
#    # otherwise, get it, so we can add labels to it
#    else:
#        textcat = nlp.get_pipe('textcat')
#
#    # add label to text classifier
#    for cat in categories:
#        textcat.add_label(str(cat))
#
#    # load the IMDB dataset
#    print("Loading IMDB data...")
#    (train_texts, train_cats), (dev_texts, dev_cats) = load_data()
#    print("examples ({} training, {} evaluation)"
#          .format(len(train_texts), len(dev_texts)))
#    train_data = list(zip(train_texts,
#                          [{'cats': cats} for cats in train_cats]))
#
#    # get names of other pipes to disable them during training
#    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
#    with nlp.disable_pipes(*other_pipes):  # only train textcat
#        optimizer = nlp.begin_training()
#        print("Training the model...")
#        print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
#        for i in range(n_iter):
#            losses = {}
#            # batch up the examples using spaCy's minibatch
#            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
#            for batch in batches:
#                texts, annotations = zip(*batch)
#                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
#                           losses=losses)
#            with textcat.model.use_params(optimizer.averages):
#                # evaluate on the dev data split off in load_data()
#                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
#            print('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
#                  .format(losses['textcat'], scores['textcat_p'],
#                          scores['textcat_r'], scores['textcat_f']))

#    # test the trained model
#    test_text = "R18.2_UAT_BDS_Contract duration and cancellation terms are not displayed inside Line back up service"
#    doc = nlp(test_text)
#    print(test_text, doc.cats)
#
#    if output_dir is not None:
#        output_dir = Path(output_dir)
#        if not output_dir.exists():
#            output_dir.mkdir()
#        nlp.to_disk(output_dir)
#        print("Saved model to", output_dir)
#
#        # test the saved model
#        print("Loading from", output_dir)
#        nlp2 = spacy.load(output_dir)
#        doc2 = nlp2(test_text)
#        print(test_text, doc2.cats)
#        
#        
#def prepare_cats_for_trained_model(nlp,new_label,train_len):
#    textcat = nlp.get_pipe('textcat')
#    textcat.add_label(new_label)
#    
#
#    
#    
#
#def prepare_cat_data(nlp:'spacy_loaded_nlp',new_label:list,train_len:'training length',model_already_trained=False):
##    gt_labels=user_story['Priority'].tolist()
#    default_dict=dict()
#    
#    if model_already_trained:
#        textcat = nlp.get_pipe('textcat')
#        cat=textcat.labels
#        cat.append(new_label)
#        for k in cat:
#            default_dict[k]=False
#    else:
#        cat =new_label
#        for k in cat:
#            default_dict[k]=False
#        
#        
#        
#    cats_list=[]
#    gt_labels=[new_label]*train_len
#    for cat in gt_labels:
#        
#        tmp_dict=default_dict.copy()
#        tmp_dict[cat]=True
#        cats_list.append(tmp_dict)
#    return cats_list
#    
#
#def load_data(limit=0, split=0.8):
#    
#    """Load data from the IMDB dataset."""
#    # Partition off part of the train data for evaluation
##    train_data, _ = thinc.extra.datasets.imdb()
##    train_data=user_story
##    random.shuffle(train_data)
##    train_data = train_data[-limit:]
#    
#    texts, labels =user_story['Summary'].values,user_story['Priority']
#    gt_labels = labels.tolist()
##    default_dict.fromkeys(categories)
#    cats_list=prepare_cat_data(gt_labels)
#    split = int(len(user_story) * split)
##    cats = [{str(y):True} for y in cats]
#    return (texts[:split], cats_list[:split]), (texts[split:], cats_list[split:])
#
#
def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

def train_cnn_for_given_label(df,filename):
    textcat=return_text_categorizer(filename)
    train_categorizer(textcat,df,filename)
    
    

if __name__=='__main__':
    
    user_story=pd.read_csv(r'D:\Testing_frameworks\Testcase-Vmops\Insight\data\interim\filtered_user_story_by_priority.csv',encoding='ISO-8859-1')
    
#    user_story=user_story.loc[user_story['Priority']!='Unprioritised']
    unprior=user_story.loc[user_story['Priority']=='Unprioritised']
#    categories=pd.factorize(user_story['Priority'])[1].tolist()
    train_cnn_for_given_label(unprior,'user_story_')

    print(predict(user_story.head()['Summary'].tolist(),'user_story'))
    
    
#    main()
    
    
#spacy.pipeline.TextCategorizer.add_label    
    
    
    
    
    
    
    
    
#ls=[]
#for txt in doc_list:
#    ls.append(doc.similarity(txt))