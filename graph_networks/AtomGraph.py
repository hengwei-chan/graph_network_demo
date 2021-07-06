#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
The class for the atom graph representation
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import CDPL.Chem as Chem

from graph_networks.utilities import *
from graph_networks.Graphs import Graph

import numpy as np
from copy import copy

class AtomGraph(Graph):
    ''' This class is the atom graph class.
    It can be utilized to represent small BasicMolecule (=ligands).
    Upon initialization, there is no calculations being done. If you
    want to generate the graph, use the call method.
    '''
    __slots__=("implicit_hydrogens","adj_matrix_edges","smiles","fingerprint",
    "adj_matrix_edges_wo","hat_adj_matrix",
    "edge_aligned_node_features","atm_dir_edge_adj_matrix","dir_edge_features")
    def __init__(self):
        super().__init__()
        self.edge_aligned_node_features = list() # node feature list based on adj degree matrix
        self.dir_edge_features = list()
            
        self.hat_adj_matrix = list() # adj_matrix + identity
        self.adj_matrix_edges = list() # adj matrix for edges (directed, but incl itself)
        self.adj_matrix_edges_wo = list() # the d-mpnn edges (directed andexcluding itself) - based on edge indices
        self.atm_dir_edge_adj_matrix = list()

        self.implicit_hydrogens = True # should hydrogens be considered as nodes as well
        
        self.smiles = list()
        self.fingerprint = None

    def __call__(self,mol,featurization='DGIN3'):
        ''' 
        PRIVATE METHOD
        generates the adjacency matrix, node and edge features of the graph \n
        Input \n
        mol (CDPL BasicMolecule): molecule the graph is based on \n
        featurization (String): Featurization for atoms and bonds. Define what kind
                    of featurization these kind of entities have. DEFAULT: 'DGIN3'.
                    Currently possible: 'DMPNN','DGIN', 'DGIN3', 'DGIN4', 'DGIN5', 'DGIN6'
                    , 'DGIN7', 'DGIN8', 'DGIN9' \n
        '''
        ### necessary for stereo
        Chem.calcAtomStereoDescriptors(mol, False)
        Chem.calcBondStereoDescriptors(mol, False)
        Chem.calcCIPPriorities(mol, True)
        Chem.calcAtomCIPConfigurations(mol, True)
        Chem.calcBondCIPConfigurations(mol, True)
        
        n_atoms = Chem.getHeavyAtomCount(mol)
        n_bonds = Chem.getHeavyBondCount(mol)
        atom_feature_dim = getFeatureDimensions(featurization,True)
        bond_feature_dim= getFeatureDimensions(featurization,False)

        atomFeatureMethod = getFeatureMethod(featurization,True)
        bondFeatureMethod = getFeatureMethod(featurization,False)

        self.adj_matrix = np.zeros((n_atoms,n_atoms))
        self.identity_matrix = np.zeros((n_atoms,n_atoms))
        self.node_features = np.zeros((n_atoms,atom_feature_dim))
        self.dir_edge_features = np.zeros((n_bonds*2,bond_feature_dim))
        self.edge_features = np.zeros((n_bonds,bond_feature_dim))
        self.adj_matrix_edges = np.zeros((n_bonds*2,n_bonds*2))
        self.adj_matrix_edges_wo = np.zeros((n_bonds*2,n_bonds*2))
        self.atm_dir_edge_adj_matrix = np.zeros((n_atoms,n_bonds*2))
        dir_edge_idx = 0
        edge_idx = 0
        edge_set_indices = list()
        sssr_list = Chem.getSSSR(mol)
        for atom in mol.atoms:
            atype = Chem.getType(atom)
            if atype == Chem.AtomType.H and self.implicit_hydrogens:
                continue
        for atom in mol.atoms:
            atm_ring_length = 0
            if Chem.getRingFlag(atom):
                for ring in sssr_list:
                    if atom in ring:
                        atm_ring_length = len(ring.atoms)
            atype = Chem.getType(atom)
            if atype == Chem.AtomType.H and self.implicit_hydrogens:
                continue
            atom_idx = mol.getAtomIndex(atom)
            self.node_features[atom_idx] = atomFeatureMethod(atom,mol,atm_ring_length)
            dir_edge_indices = list()
            for other_atom in mol.atoms:
                other_atype = Chem.getType(other_atom)
                if other_atype == Chem.AtomType.H and self.implicit_hydrogens:
                    continue
                other_atom_idx = mol.getAtomIndex(other_atom)
                if atom_idx == other_atom_idx: self.identity_matrix[atom_idx][atom_idx] = 1
                if mol.getAtom(atom_idx).findBondToAtom(mol.getAtom(other_atom_idx)):
                    bnd = mol.getAtom(atom_idx).findBondToAtom(mol.getAtom(other_atom_idx))
                    self.adj_matrix[atom_idx][other_atom_idx] = 1
                    bond_ring_length = 0
                    if Chem.getRingFlag(bnd):
                        for ring in sssr_list:
                            if bnd in ring:
                                bond_ring_length = len(ring.atoms)
                    self.dir_edge_features[dir_edge_idx] = bondFeatureMethod(bnd,mol,bond_ring_length)
                    edge_set = frozenset((atom_idx,other_atom_idx))
                    if not edge_set in edge_set_indices:
                        edge_set_indices.append(edge_set)
                        self.edge_features[edge_idx] = bondFeatureMethod(bnd,mol,bond_ring_length)
                        edge_idx = edge_idx + 1
                    dir_edge_indices.append(dir_edge_idx)
                    dir_edge_idx = dir_edge_idx + 1
            for i in range(len(dir_edge_indices)):
                for edge_idx_counter in dir_edge_indices:
                    self.adj_matrix_edges[dir_edge_indices[i]][edge_idx_counter] = 1
                    self.atm_dir_edge_adj_matrix[atom_idx][edge_idx_counter] = 1
                dir_edge_indices_wo = [x for j,x in enumerate(dir_edge_indices) if j!=i]
                for edge_idx_counter in dir_edge_indices_wo:
                    self.adj_matrix_edges_wo[dir_edge_indices[i]][edge_idx_counter] = 1

        self.hat_adj_matrix = self.identity_matrix+self.adj_matrix
        self.norm_adj_matrix = np.diag(np.sum(self.hat_adj_matrix, axis=1)**(-0.5))
        degree_array = np.sum(self.atm_dir_edge_adj_matrix, axis=1)
        for i,k in enumerate(degree_array):
            inter = np.repeat([self.node_features[i]],
                int(k),axis=0)
            self.edge_aligned_node_features.extend(inter)


    def getIndex(self):
        return self.index

    def getAtomGraphNeighbors(self):
        return list(set(self.atom_graph_neighbors))

    def setAtomGraphNeighbors(self,neighbors):
        self.atom_graph_neighbors = neighbors

    def appendAtomGraphNeighbors(self, neighbor):
        self.atom_graph_neighbors.append(neighbor)

    def setSmiles(self, smiles):
        self.smiles = smiles

    def getSmiles(self):
        return self.smiles

    def setFragmentClass(self, fragment_class):
        self.fragment_class = fragment_class

    def getFragmentClass(self):
        return self.fragment_class

    def setFingerprint(self, fingerprint):
        self.fingerprint = fingerprint

    def getFingerprint(self):
        return self.fingerprint

    def setProperty(self,property_string,property_):
        '''
        Input \n
        property_name (string): the property name the property string should be saved
        under.\n
        property_ (string): the "real" property, that should be saved.
        '''
        self.properties[property_string] = property_

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

    def getName(self):
        return self.name

    def setName(self,name):
        self.name = name