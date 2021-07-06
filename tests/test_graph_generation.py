"""
test for the graph_networks package and graph generation.
"""

# Import package, test suite, and other packages as needed
import graph_networks
import pytest
import sys
import os

def test_utilities_readChemblXls():
    import graph_networks.utilities as ut
    print(os.getcwd(),"test")
    data = ut.readChemblXls("./data/CHE_3.xls")
    assert("CHEMBL400569" == data[0][0])
    assert("CCNCc1cncc(c1)c2cnc3[nH]nc(c4nc5cc(ccc5[nH]4)N6CCN(C)CC6)c3c2" == data[0][1])

def testt_utilities_generateFromSmiles():
    import graph_networks.utilities as ut
    path = "./data/twoD.smi"
    mol = ut.CDPLmolFromSmiles(path,False)
    assert(len(mol.atoms) == 22) #2D
    mol_conf = ut.CDPLmolFromSmiles(path,True)
    assert(len(mol_conf.atoms) == 37) #3D

def testt_utilities_generateFromSmiles():
    import graph_networks.utilities as ut

    data = ut.readChemblXls("./data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[2][1],False) # same smiles then twoD.smi
    mol_conf =ut.CDPLmolFromSmiles(data[2][1],True) # same smiles then twoD.smi

    assert(len(mol.atoms) == 22) #2D
    assert(len(mol_conf.atoms) == 37) #3D with Hydrogens

def testt_utilities_generateFromSDF():
    import graph_networks.utilities as ut
    path = "./data/threeD_activity.sdf"
    mol = ut.CDPLmolFromSdf(path,False)
    mol_conf = ut.CDPLmolFromSdf(path,True)
    assert(len(mol.atoms) == 37) # 3D with hydrogens
    assert(len(mol_conf.atoms) == 37) #3D with Hydrogens, but different conf.

def test_atomgraph_featurizations1():
    import graph_networks.utilities as ut
    from graph_networks.AtomGraph import AtomGraph
    data = ut.readChemblXls("./data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[0][1],True)
    feature_type = 'DMPNN'
    ag = AtomGraph()
    ag.__call__(mol,feature_type)
    assert len(ag.dir_edge_features[0]) == 12
    assert ag.dir_edge_features[0][0] == 1.0
    assert ag.dir_edge_features[8][0] == 0.0
    assert ag.dir_edge_features[8][1] == 1.0
    assert len(ag.node_features[0]) == 33

def test_atomgraph_featurizations5():
    import graph_networks.utilities as ut
    from graph_networks.AtomGraph import AtomGraph
    data = ut.readChemblXls("./data/CHE_3.xls")
    mol =ut.CDPLmolFromSmiles(data[0][1],True)
    feature_type = 'DGIN5'
    ag = AtomGraph()
    ag.__call__(mol,feature_type)
    assert len(ag.dir_edge_features[0]) == 3
    assert ag.dir_edge_features[0][0] == 1.0
    assert ag.dir_edge_features[8][0] == 0.0
    assert ag.dir_edge_features[8][1] == 1.0
    assert len(ag.node_features[0]) == 16

