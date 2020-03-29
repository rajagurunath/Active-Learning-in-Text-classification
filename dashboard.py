# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 21:48:27 2018

@author: Dell
"""

#class sklearnBoard():
#    pass

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from datetime import datetime as dt
import plotly.graph_objs as go
import dash_auth
# import plotly.dashboard_objs as dashboard

class Dashboard():
    """
    prepares grid with specified number of columns in a single row and returns the div element
    
    input : pass respective number of html elements to the respective Function:
    
    output :returns grid with specified number of columns
    
    """
    def __init__(self):
        pass
#        self.no_of_grid=no_of_grid
    def three_columns_grid(self,element1,element2,element3,className='four columns'):
        """
        input :Three html elements (from Dash)
        output: returns dash html div with three grids
        """
        three_colum_div=html.Div(children=[html.Div(element1,className=className),html.Div(element2,className=className),
                               html.Div(element3,className=className)],className="row")
        return three_colum_div
    def two_columns_grid(self,element1,element2,className='six columns'):
        """
        input :Two html elements (from Dash)
        output: returns dash html div with two grids
        """
        two_colum_div=html.Div(children=[html.Div(children=element1,className=className),html.Div(children=element2,className=className)],
                               className="row")
        return two_colum_div
    
#class Univariate():
#    """
#    
#    """
#    def pie_chart():
#        


