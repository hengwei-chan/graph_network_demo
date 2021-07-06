#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
The super class for all graph representations
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import CDPL.Chem as Chem

from graph_networks.utilities import *

import numpy as np
from copy import copy

class Graph():
    ''' 
    This class is the super graph class.
    It is a template for different graph classes.
    It contains the methods for the different featurizations.
    '''
    __slots__=("node_features","edge_features","properties","name","adj_matrix",
    "norm_adj_matrix","degree_matrix","distance_matrix","identity_matrix")
    def __init__(self,all_connected=False):
        self.node_features = list() # list of node_features
        self.edge_features = list() # list of edge_features
            
        self.properties = dict()
        self.name = None # of the current compound. should be unique

        self.adj_matrix = list() # adj matrix of the graph
        self.degree_matrix = list() # degree matrix of the graph
        self.identity_matrix = list() # idt matrix of the graph

        self.distance_matrix = list() # distance matrix of the graph

    def __call__(self,mol,calc_frag_contrib=False):
        '''
        Should be overwritten - used to generate the graph.
        '''

    def setProperty(self,property_string,property_):
        '''
        Input \n
        property_name (string): the property name the property string should be saved
        under.\n
        property_ (string): the "real" property, that should be saved.
        '''
        self.properties[property_string] = property_


    def getProperty(self,property_string):
        '''
        Input \n
        property_name (string): Demanded property name \n
        Returns \n
        (string): containing the "property_name" property with property_name \n
        (ValueError): graph does not contain "property_name"
        '''
        if property_string in self.properties.keys():
            return self.properties[property_string]
        else:
            raise ValueError("Property name",property_string,"does not exist. Only those names are present:",self.properties.keys())
    
    def setProperties(self,property_list,property_,property_type=None):
        '''
        Input \n
        property_list (list of strings): the property names the property strings should be saved
        under.\n
        property_ (list of string): the "real" properties, that should be saved. \n
        property_type (list of string): DEFAULT NONE; a list of property types.
        '''
        prop_iter = 0
        for prop in property_list:
            if not property_[prop_iter]:
                self.properties[prop] = None
                prop_iter += 1
                continue
            self.properties[prop] = property_type[prop_iter](property_[prop_iter])
            prop_iter += 1
        
    
    def getProperties(self):
        '''
        Returns \n
        (dict): containing all properties of the graph defined \n
        '''
        return self.properties

    def getName(self):
        return self.name

    def setName(self,name):
        self.name = name
