# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 12:04:33 2018

@author: gurunath.lv
"""

import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import  plotly.graph_objs as go

import numpy as np
import pandas as pd
import json
from tf_universal_sent_emb import get_similar_records
from spacy_text_classifier_cnn import train_cnn_for_given_label,predict
import glob
import os
from custom_classifier import customKNN,ParagraphVectors

from dashboard import Dashboard
from flask import Flask
import flask

#import glob
from sklearn.pipeline import Pipeline
import pickle
from lime.lime_text import LimeTextExplainer



server = Flask(__name__)


app=dash.Dash(name = __name__, server = server)
if not os.path.exists('tmp'):
    os.mkdir('tmp')
DIRECTORY_PATH=r'tmp\\'


#app = dash.Dash()

app.scripts.config.serve_locally = True

app.config['suppress_callback_exceptions']=True

custom_dush=Dashboard()

#prodapt=html.Div(html.Img(src='http://www.prodapt.com/wp-content/uploads/logo_prodapt.png')),
#vs=html.H1('vs')
#reinfer=html.Div(html.Img(src='https://d1qb2nb5cznatu.cloudfront.net/startups/i/703763-82fa920eed7d56e7cdcee1b1d9a30b14-medium_jpg.jpg?buster=1440002957')),


#logo=custom_dush.three_columns_grid(prodapt,vs,reinfer)

app.layout = html.Div([
#    logo,
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'}),
    dcc.Input(id='user-input-for-similarity', 
                                       value='Enter the sentence', type='text',
                                       style={'width': '49%','align':'center'}),
    
    html.Div(id='similar-docs'),
    html.Br(),
    html.H3('Training dataset'),
    html.Div(id='output'),
    html.Button('Train',id='train-button'),
    html.Br(),
    dcc.Input(id='user-input-for-prediction', 
                                       value='Enter the sentence to predict', type='text',
                                       style={'width': '49%','align':'center'}),
    html.H1(id='train-output'),
#    html.Button('Del data',id='delete-button'),
    html.H1(id='del-output'),
    dcc.Graph(id='predict-output'),
    html.Br(),
    dcc.Link('Why ML made this Prediction !!!!', href='/explain'),
             
    
#    html.Div([
#        html.Pre(id='output', className='two columns'),
#        html.Div(
#            dcc.Graph(
#                id='graph',
#                style={
#                    'overflow-x': 'wordwrap'
#                }
#            ),
#            className='ten columns'
#        )
#    ], className='row')
])
    



def parse_contents(contents, file_name, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    global df 
    global filename
    
    try:
        if 'csv' in file_name:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf8')))
        elif 'xls' in file_name:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    """
    pass similar contents df not file uploaded df 
    To be filled
    
    """
#    df.to_csv(r'{}'.format(filename),index=False)
    filename=file_name
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # Use the DataTable prototype component:
        # github.com/plotly/dash-table-experiments
        dt.DataTable(rows=df.to_dict('records'),id='edit-table'),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
#        html.Div('Raw Content'),
#        html.Pre(contents[0:200] + '...', style={
#            'whiteSpace': 'pre-wrap',
#            'wordBreak': 'break-all'
#        })
    ])

@app.callback(
    Output(component_id='similar-docs', component_property='children'),
    [Input(component_id='user-input-for-similarity', component_property='value')])
def get_similar_docs(sent):
    print('similar docs called ',sent)
#    path=glob.glob(r'*.csv')
#    df=pd.read_csv(path[0],encoding='ISO-8859-1')
    global similar_df
    similar_series=get_similar_records(sent,df[df.columns[0]])
    similar_df=pd.DataFrame(columns=['Similar_sentences','labels'])
    similar_df['Similar_sentences']=similar_series
    
    
    print('check',similar_df.head())
#    similar_df.to_csv
    return html.Div(dt.DataTable(rows=similar_df.to_dict('records'),id='edit-table-similar'),)
    


def train_custom_classifier(similar_df,filename):
    texts, labels =similar_df.iloc[:,0].values,similar_df.iloc[:,1].values
    print(type(labels),type(texts),labels)
    if glob.glob(r'{}{}_knn_centroid.pkl'.format(DIRECTORY_PATH,filename)):
          
        dict_=pickle.load(open(glob.glob(r'{}*{}_knn_centroid.pkl'.format(DIRECTORY_PATH,filename))[0],'rb'))
		
        pipe=Pipeline(steps=[('pv',ParagraphVectors(filename=filename)),('knn',customKNN(label_to_vect_dict=dict_))])
        
        label_encoding=pickle.load(open(glob.glob(r'{}*{}_label_encoding.pkl'.format(DIRECTORY_PATH,filename))[0],'rb'))
        
        for idx,lab in enumerate(set(labels)):
             label_encoding[idx+len(label_encoding)]=lab
    else:
        label_encoding=dict()
        pipe=Pipeline(steps=[('pv',ParagraphVectors(filename=filename)),('knn',customKNN())])
        for idx,lab in enumerate(set(labels)):
            label_encoding[idx]=lab
    look_up=dict()
    for k,v in label_encoding.items():
        look_up[v]=k
    pipe.fit(texts,pd.Series(labels).map(look_up))
    dict_=pipe.named_steps.knn.get_centroid()
    pickle.dump(dict_,open(r'{}{}_knn_centroid.pkl'.format(DIRECTORY_PATH,filename),'wb'))
    pickle.dump(label_encoding,open(r'{}{}_label_encoding.pkl'.format(DIRECTORY_PATH,filename),'wb'))
    
#train_custom_classifier(user_story,filename)
            
def explain_prediction(sent,pipe,filename):
#    vect=transform_inp_sent_to_vect(sent)
    
    label_encoding=pickle.load(open(glob.glob(r'{}{}_label_encoding.pkl'.format(DIRECTORY_PATH,filename))[0],'rb'))
    labels=list(label_encoding.values())
    explainer = LimeTextExplainer(class_names=labels)
    
    exp = explainer.explain_instance(sent, pipe.predict_proba,labels=labels)
    return exp.save_to_file(r'{}explanation.html'.format(DIRECTORY_PATH))

def predict_custom_classifier(sent_list,filename):
    print('inside fn',filename)
    dict_=pickle.load(open(glob.glob(r'{}{}_knn_centroid.pkl'.format(DIRECTORY_PATH,filename))[0],'rb'))
    pipe=Pipeline(steps=[('pv',ParagraphVectors(filename=filename)),('knn',customKNN(label_to_vect_dict=dict_))])
    label_encoding=pickle.load(open(glob.glob(r'{}{}_label_encoding.pkl'.format(DIRECTORY_PATH,filename))[0],'rb'))
    
#    pred_dict=label_encoding.copy()
    
    pred=[]
    for sent in sent_list:
        explain_prediction(sent,pipe,filename)
        pred.append(pipe.predict_proba(sent))
    
    return pred,list(label_encoding.values())
	

    
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents'),
               Input('upload-data', 'filename'),
               Input('upload-data', 'last_modified')])
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        
        return children


def prediction_bar_chart(xaxis,labels):
    trace1 = go.Bar(
                x = xaxis,
                y = labels,
                text= labels,
        marker=dict(
        color='rgb(158,202,225)',
        line=dict(
            color='rgb(8,48,107)',
            width=1.5,
        )
    ),
        orientation='h',
        )
    layout = go.Layout(
        title='probabilities',
        xaxis=dict(title='probability',),
#        yaxis=dict(title='priority')
        )

    fig=go.Figure(data=[trace1],layout=layout)
    return fig





@app.callback(
    Output('train-output', 'children'),
    [Input('train-button', 'n_clicks')])
def call_spacy_model_training(n_clicks):
#    print('button clicks:',n_clicks)
    if n_clicks!=None:
#        path=glob.glob(r'train.csv')
#        dff=pd.read_csv(path[0],encoding='ISO-8859-1')
    #    filename=path[0].split('.')[0].split('_')[0]
#        filename=path[0].split('\\')[-1].split('_train_data')[0]
        train_custom_classifier(similar_df,filename)
        return 'custom classifier trained'



@app.server.route('/explain')
def upload_file():
   return flask.send_from_directory(DIRECTORY_PATH,'explanation.html')



#@app.callback(
#    Output('del-output', 'children'),
#    [Input('delete-button', 'n_clicks')])
#def delete_data_tmp_folder(n):
#    print('delete:',n)
#    path=glob.glob(r'D:\Testing_frameworks\Testcase-Vmops\Insight\src\features\tmp\*.csv')
#    os.remove(path[0])
#    return 'data deleted'


@app.callback(
    Output('predict-output', 'figure'),
    [Input('user-input-for-prediction', component_property='value')])
def predict_cat(sent):
    print(sent)
#    path=glob.glob(r'D:\Testing_frameworks\Testcase-Vmops\Insight\src\features\tmp\*.csv')
#    filename=path[0].split('\\')[-1].split('.')[0]
    pred,labels=predict_custom_classifier([sent],filename)
#    print(dict_)
    print(pred,labels)
    return prediction_bar_chart(list(np.absolute(pred[0].ravel())),labels)             #json.dumps(predict([sent],filename)[0],indent=2)

def generate_table(dataframe, max_rows=10):
    return (
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


@app.callback(
    Output('output', 'children'),
    [Input('edit-table-similar', 'rows')])
def update_selected_row_indices(rows):
#    path=glob.glob(r'D:\Testing_frameworks\Testcase-Vmops\Insight\src\features\tmp\*.csv')
#    filename=path[0].split('\\')[-1].split('.')[0]
    df=pd.DataFrame(rows)
#    print(df)
    global similar_df
    similar_df=df
#    df.to_csv(r'D:\Testing_frameworks\Testcase-Vmops\Insight\src\features\tmp\{}_train_data.csv'.format(filename),index=False)
    return generate_table(df)


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/brPBPO.css"})


if __name__ == '__main__':
    app.run_server(debug=True,port =8050)