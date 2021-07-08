#!/usr/bin/env python

# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Different helper functions for
1. graph generation including
    a. featurization of vertices and edges
2. 
"""

# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import logging
log = logging.getLogger(__name__)

import CDPL.Chem as Chem
import CDPL.Base as Base
import CDPL.Biomol as Biomol
import CDPL.ConfGen as ConfGen

import tensorflow as tf

import numpy as np 
import pickle
import os
import xlrd
from scipy import stats
import argparse

# =============================================================================
# GLOBAL FIELDS
# =============================================================================

# The elements used for the atom featurization
ELEM_LIST =[1, 6, 7, 8, 16, 9, 15, 17, 35, 53,"unknown"] #H;C;N;O;Si;F;P;CL;BR;I

############# the dimensions for the different featurizations
########
DMPNN_ATOM_FEATURE_DIM = 22+len(ELEM_LIST) # 11 + 22 = 33
DMPNN_EDGE_FEATURE_DIM = 12
ALL_DPMNN_FEATURE_DIM = DMPNN_EDGE_FEATURE_DIM+DMPNN_ATOM_FEATURE_DIM #45

######## 
DGIN2_ATOM_FEATURE_DIM = 31+len(ELEM_LIST) # 11 + 31 = 42
DGIN2_EDGE_FEATURE_DIM = 22
ALL_DGIN2_FEATURE_DIM = DGIN2_EDGE_FEATURE_DIM+DGIN2_ATOM_FEATURE_DIM #64

######## 
DGIN3_ATOM_FEATURE_DIM = 24+len(ELEM_LIST) # 11 + 24 = 35
DGIN3_EDGE_FEATURE_DIM = 15
ALL_DGIN3_FEATURE_DIM = DGIN3_EDGE_FEATURE_DIM+DGIN3_ATOM_FEATURE_DIM #50

######## 
DGIN4_ATOM_FEATURE_DIM = 33+len(ELEM_LIST) # 11 + 31 = 42
DGIN4_EDGE_FEATURE_DIM = 24
ALL_DGIN4_FEATURE_DIM = DGIN4_EDGE_FEATURE_DIM+DGIN4_ATOM_FEATURE_DIM #68

######## 
DGIN5_ATOM_FEATURE_DIM = 5+len(ELEM_LIST) # 11 + 5 = 16
DGIN5_EDGE_FEATURE_DIM = 3
ALL_DGIN5_FEATURE_DIM = DGIN5_EDGE_FEATURE_DIM+DGIN5_ATOM_FEATURE_DIM #19

######## 
DGIN6_ATOM_FEATURE_DIM = 5+len(ELEM_LIST) # 11 + 5 = 16
DGIN6_EDGE_FEATURE_DIM = 24
ALL_DGIN6_FEATURE_DIM = DGIN6_EDGE_FEATURE_DIM+DGIN6_ATOM_FEATURE_DIM #40

######## 
DGIN7_ATOM_FEATURE_DIM = 33+len(ELEM_LIST) # 11 + 31 = 42
DGIN7_EDGE_FEATURE_DIM = 3
ALL_DGIN7_FEATURE_DIM = DGIN7_EDGE_FEATURE_DIM+DGIN7_ATOM_FEATURE_DIM #45

######## 
DGIN8_ATOM_FEATURE_DIM = 20 + len(ELEM_LIST) # 31
DGIN8_EDGE_FEATURE_DIM = 14
ALL_DGIN8_FEATURE_DIM = DGIN8_EDGE_FEATURE_DIM+DGIN8_ATOM_FEATURE_DIM #45

######## 
DGIN9_ATOM_FEATURE_DIM = 20+ len(ELEM_LIST) # 31
DGIN9_EDGE_FEATURE_DIM = 11
ALL_DGIN9_FEATURE_DIM = DGIN9_EDGE_FEATURE_DIM+DGIN9_ATOM_FEATURE_DIM #22


############# used for cleaning molecule libraries.
########
MST_MAX_WEIGHT = 100 

REMOVE_FLUORINATED = True
FLUOR_ATOM_COUNT = 9
MIN_HEAVY_ATOM_COUNT = 0
VALID_ATOM_TYPES = [Chem.AtomType.H, Chem.AtomType.C, Chem.AtomType.F, Chem.AtomType.Cl, Chem.AtomType.Br, Chem.AtomType.I, Chem.AtomType.N, Chem.AtomType.O, Chem.AtomType.S, Chem.AtomType.Se, Chem.AtomType.P, Chem.AtomType.Pt, Chem.AtomType.As, Chem.AtomType.Si]    
CARBON_ATOMS_MANDATORY = True
NEUTRALIZE = True
KEEP_ONLY_LARGEST_COMP = True 

LOG_LEVELS = {
    0: logging.CRITICAL,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
    }



# =============================================================================
# Chemistry specific Methods
# =============================================================================

def _isConjugated(bond,mol):
        """
        PRIVATE METHOD 
        Checks whether the target bond is conjugated or not (DB with DB) \n
        Arguments: \n
            bond {[CDPL BasicBond]} -- [target bond] \n
            bond {[CDPL BasicMolecule]} -- [molecule, the bond is in] \n
        Return: \n
            {[Boolean]} -- [True or False]    
        """
        conj = False

        if Chem.getOrder(bond) != 1:
            for nbr in bond.atoms[0].atoms:
                if nbr.getIndex() != bond.atoms[1].getIndex() and Chem.isUnsaturated(nbr, mol):
                    conj = True
                    break

            if not conj:
                for nbr in bond.atoms[1].atoms:
                    if nbr.getIndex() != bond.atoms[0].getIndex() and Chem.isUnsaturated(nbr, mol):
                        conj = True
                        break

        return conj

def _CDPLgenerateConformation(cdplMol):
    '''
    PRIVAT METHOD
    generates one CDPL Molecule conformation. \n
    Input: \n
    mol (CDPL BasicMolecule): a CDPL BasicMolecule \n
    Return: \n
    (CDPL BasicMolecule): the corresponding random conf. for the input BasicMolecule
     '''
    _CDPLconfigForConformation(cdplMol)

    ConfGen.prepareForConformerGeneration(cdplMol)
    cg = ConfGen.ConformerGenerator()
    cg.generate(cdplMol)
    coords = cg.getConformer(0) # uses currently only one conformation
    Chem.set3DCoordinates(cdplMol, coords)

    return cdplMol

def _CDPLconfigForConformation(mol):
    '''
    PRIVAT METHOD
    configures a CDPL Molecule for conformation generation. \n
    Input: \n
    mol (CDPL BasicMolecule): a CDPL BasicMolecule \n
    Return: \n
    (CDPL BasicMolecule): the configured input BasicMolecule
     '''
    Chem.perceiveComponents(mol, False)
    Chem.perceiveSSSR(mol, True)
    Chem.setRingFlags(mol, False)
    Chem.calcImplicitHydrogenCounts(mol, False)
    Chem.perceiveHybridizationStates(mol, False)
    Chem.setAromaticityFlags(mol, False)
    Chem.calcCIPPriorities(mol, False)
    Chem.calcAtomCIPConfigurations(mol, False)
    Chem.calcBondCIPConfigurations(mol, False)
    Chem.calcAtomStereoDescriptors(mol, False)
    Chem.calcBondStereoDescriptors(mol, False)
    Chem.calcTopologicalDistanceMatrix(mol, False)

    Chem.generate2DCoordinates(mol, False)
    Chem.generateBond2DStereoFlags(mol, True)

def _cleanMolecule(mol):
    '''
    PRIVATE METHOD \n
    This method cleans a CDPL molecule
    '''
    Chem.perceiveComponents(mol, False)
    Chem.perceiveSSSR(mol, False)
    Chem.setRingFlags(mol, False)
    Chem.calcImplicitHydrogenCounts(mol, False)
    Chem.perceiveHybridizationStates(mol, False)
    Chem.setAromaticityFlags(mol, False)

    modified = False
    comps = Chem.getComponents(mol)

    if comps.getSize() > 1 and KEEP_ONLY_LARGEST_COMP:
        largest_comp = None

        for comp in comps:
            if largest_comp is None:
                largest_comp = comp
            elif comp.getNumAtoms() > largest_comp.getNumAtoms():
                largest_comp = comp

        Chem.perceiveComponents(largest_comp, False)
        Chem.perceiveSSSR(largest_comp, False)
        Chem.setName(largest_comp, Chem.getName(mol))

        modified = True

        if Chem.hasName(mol):
            Chem.setName(largest_comp, Chem.getName(mol))

        if Chem.hasStructureData(mol):
            Chem.setStructureData(largest_comp, Chem.getStructureData(mol))

        mol = largest_comp

    if Chem.getHeavyAtomCount(mol) < MIN_HEAVY_ATOM_COUNT:
        return None

    if REMOVE_FLUORINATED and Chem.getAtomCount(mol, Chem.AtomType.F) > FLUOR_ATOM_COUNT:
        return None

    carbon_seen = False
    hs_to_remove = list()

    for atom in mol.atoms:
        atom_type = Chem.getType(atom)
        invalid_type = True

        for valid_type in VALID_ATOM_TYPES:
            if Chem.atomTypesMatch(valid_type, atom_type):
                invalid_type = False
                break

        if invalid_type:
            return None

        if atom_type == Chem.AtomType.C:
            carbon_seen = True

        if NEUTRALIZE:
            form_charge = Chem.getFormalCharge(atom)

            if form_charge != 0:
                for nbr_atom in atom.atoms:
                    if Chem.getFormalCharge(nbr_atom) != 0:
                        form_charge = 0
                        break

            if form_charge != 0:
                if form_charge > 0:
                    form_charge -= Chem.getImplicitHydrogenCount(atom)

                    if form_charge < 0:
                        form_charge = 0

                    for nbr_atom in atom.atoms: 
                        if form_charge == 0:
                            break

                        if Chem.getType(nbr_atom) == Chem.AtomType.H:
                            hs_to_remove.append(nbr_atom)
                            form_charge -= 1
                        
                    Chem.setFormalCharge(atom, form_charge)

                else:
                    Chem.setFormalCharge(atom, 0)
                            
                modified = True
    
    if CARBON_ATOMS_MANDATORY and carbon_seen == False:
        return None

    if len(hs_to_remove) > 0:
        for atom in hs_to_remove:
            mol.removeAtom(mol.getAtomIndex(atom))
 
        for atom in mol.atoms:
            Chem.setImplicitHydrogenCount(atom, Chem.calcImplicitHydrogenCount(atom, mol))
            Chem.setHybridizationState(atom, Chem.perceiveHybridizationState(atom, mol))

    # if modified:
    #     return None
        # stats.modified += 1

    return mol


# =============================================================================
# Featurization Methods
# =============================================================================

def getFeatureDimensions(feature_type='DGIN3',atom=True):
    ''' 
    Returns the dimensions for a particular featurization type. \n
    Input \n
    feature_type (String): Featurization method for atoms and bonds. Define what kind
                    of feature_type these kind of entities have. DEFAULT: 'DGIN3'.
                    Currently possible: 'DMPNN','DGIN', 'DGIN3', 'DGIN4', 'DGIN5', 'DGIN6'
                    , 'DGIN7', 'DGIN8', 'DGIN9'  \n
    atom (bool): Are the methods for atoms - DEFAULT: True, if not for atoms, enter False \n
    Return \n
    (int): the corresponding feature dimension
    '''
    switcher = {
        "DMPNN_atom":DMPNN_ATOM_FEATURE_DIM,
        "DGIN_atom":DMPNN_ATOM_FEATURE_DIM,
        "DGIN2_atom":DGIN2_ATOM_FEATURE_DIM,
        "DGIN3_atom":DGIN3_ATOM_FEATURE_DIM,
        "DGIN4_atom":DGIN4_ATOM_FEATURE_DIM,
        "DGIN5_atom":DGIN5_ATOM_FEATURE_DIM,
        "DGIN6_atom":DGIN6_ATOM_FEATURE_DIM,
        "DGIN7_atom":DGIN7_ATOM_FEATURE_DIM,
        "DGIN8_atom":DGIN8_ATOM_FEATURE_DIM,
        "DGIN9_atom":DGIN9_ATOM_FEATURE_DIM,
        "DMPNN_bond":DMPNN_EDGE_FEATURE_DIM,
        "DGIN_bond":DMPNN_EDGE_FEATURE_DIM,
        "DGIN2_bond":DGIN2_EDGE_FEATURE_DIM,
        "DGIN3_bond":DGIN3_EDGE_FEATURE_DIM,
        "DGIN4_bond":DGIN4_EDGE_FEATURE_DIM,
        "DGIN5_bond":DGIN5_EDGE_FEATURE_DIM,
        "DGIN6_bond":DGIN6_EDGE_FEATURE_DIM,
        "DGIN7_bond":DGIN7_EDGE_FEATURE_DIM,
        "DGIN8_bond":DGIN8_EDGE_FEATURE_DIM,
        "DGIN9_bond":DGIN9_EDGE_FEATURE_DIM
    }

    return switcher.get(feature_type+('_atom' if atom else '_bond'), lambda: "Invalid featurization type!")




def getFeatureMethod(feature_type='DGIN3',atom=True):
    ''' 
    Returns the method for a particular featurization type. \n
    Input \n
    feature_type (String): Featurization method for atoms and bonds. Define what kind
                    of feature_type these kind of entities have. DEFAULT: 'DGIN3'.
                    Currently possible: 'DMPNN','DGIN', 'DGIN3', 'DGIN4', 'DGIN5', 'DGIN6'
                    , 'DGIN7', 'DGIN8', 'DGIN9'  \n
    atom (bool): Are the methods for atoms - DEFAULT: True, if not for atoms, enter False \n
    Return \n
    (method) The corresponding featurization method. 
    '''
    switcher = {
        "DMPNN_atom":atomFeaturesDMPNN,
        "DGIN_atom":atomFeaturesDGIN,
        "DGIN2_atom":atomFeaturesDGIN2,
        "DGIN3_atom":atomFeaturesDGIN3,
        "DGIN4_atom":atomFeaturesDGIN4,
        "DGIN5_atom":atomFeaturesDGIN5,
        "DGIN6_atom":atomFeaturesDGIN6,
        "DGIN7_atom":atomFeaturesDGIN7,
        "DGIN8_atom":atomFeaturesDGIN8,
        "DGIN9_atom":atomFeaturesDGIN9,
        "DMPNN_bond":bondFeaturesDMPNN,
        "DGIN_bond":bondFeaturesDGIN,
        "DGIN2_bond":bondFeaturesDGIN2,
        "DGIN3_bond":bondFeaturesDGIN3,
        "DGIN4_bond":bondFeaturesDGIN4,
        "DGIN5_bond":bondFeaturesDGIN5,
        "DGIN6_bond":bondFeaturesDGIN6,
        "DGIN7_bond":bondFeaturesDGIN7,
        "DGIN8_bond":bondFeaturesDGIN8,
        "DGIN9_bond":bondFeaturesDGIN9
    }

    return switcher.get(feature_type+('_atom' if atom else '_bond'), lambda: "Invalid featurization type!")


#########
# atom features

def atomFeaturesDMPNN(atom,mol,atm_ring_length=0):
        ''' 
        generates the atom features as in Yang et al. \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0]) 
                +_getAllowedSet(Chem.getCIPConfiguration(atom), [8,2,4,0])
                    +_getAllowedSet(Chem.getHybridizationState(atom), [1,2,3,7,8,1])
                +list([float(Chem.getAromaticityFlag(atom))])
                +_getAllowedSet(Chem.getImplicitHydrogenCount(atom), [0,1,2,3,4,5])))

def atomFeaturesDGIN(atom,mol,atm_ring_length=0):
        ''' 
        generates the atom features of the DGIN 1 - e.g. as in model2_4_6 \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0]) 
                +_getAllowedSet(Chem.getCIPConfiguration(atom), [8,2,4,0])
                    +_getAllowedSet(Chem.getHybridizationState(atom), [1,2,3,7,8,1])
                +list([float(Chem.getAromaticityFlag(atom))])
                +_getAllowedSet(Chem.getImplicitHydrogenCount(atom), [0,1,2,3,4,5])))

def atomFeaturesDGIN2(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 2 - approach with e.g. rot bond and others \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0]) 
                +_getAllowedSet(Chem.getCIPConfiguration(atom), [8,2,4,0])
                    +_getAllowedSet(Chem.getHybridizationState(atom), [1,2,3,7,8,1])
                +list([float(Chem.isAmideCenterAtom(atom,mol))])
                +list([float(Chem.getAromaticityFlag(atom))])
                +_getAllowedSet(atm_ring_length, [0,3,4,5,6,7,8,9])
                +_getAllowedSet(Chem.getImplicitHydrogenCount(atom), [0,1,2,3,4,5])))

def atomFeaturesDGIN3(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN -novel approach with e.g. rot bond and others \n
        differs to DGIN2 in that a different ring_length counter (1/ring_leng) is used instead of allowed set\n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST)) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0])
                +_getAllowedSet(Chem.getCIPConfiguration(atom), [8,2,4,0])
                    +_getAllowedSet(Chem.getHybridizationState(atom), [1,2,3,7,8,1])
                +list([float(Chem.isAmideCenterAtom(atom,mol))])
                +list([float(Chem.getAromaticityFlag(atom))])
                +list([float(0.0) if atm_ring_length == float(0.0) else float(1/atm_ring_length)])
                +_getAllowedSet(Chem.getImplicitHydrogenCount(atom), [0,1,2,3,4,5]))

def atomFeaturesDGIN4(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 4 - approach with e.g. rot bond and others \n
        differs to DGIN2 in that a different ring_length counter is used \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0]) 
                +_getAllowedSet(Chem.getCIPConfiguration(atom), [8,2,4,0])
                    +_getAllowedSet(Chem.getHybridizationState(atom), [1,2,3,7,8,1])
                # +list([float(Chem.getStereoCenterFlag(atom))])
                +list([float(Chem.isAmideCenterAtom(atom,mol))])
                +list([float(Chem.getAromaticityFlag(atom))])
                +_getAllowedSet(atm_ring_length, [0,3,4,5,6,7,8,9,10,11])
                +_getAllowedSet(Chem.getImplicitHydrogenCount(atom), [0,1,2,3,4,5])))

def atomFeaturesDGIN5(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 5 - basic input \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0])))

def atomFeaturesDGIN6(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 6 - basic input \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0])))

def atomFeaturesDGIN7(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 7 - approach with e.g. rot bond and others \n
        differs to DGIN2 in that a different ring_length counter is used \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0]) 
                +_getAllowedSet(Chem.getCIPConfiguration(atom), [8,2,4,0])
                    +_getAllowedSet(Chem.getHybridizationState(atom), [1,2,3,7,8,1])
                # +list([float(Chem.getStereoCenterFlag(atom))])
                +list([float(Chem.isAmideCenterAtom(atom,mol))])
                +list([float(Chem.getAromaticityFlag(atom))])
                +_getAllowedSet(atm_ring_length, [0,3,4,5,6,7,8,9,10,11])
                +_getAllowedSet(Chem.getImplicitHydrogenCount(atom), [0,1,2,3,4,5])))

def atomFeaturesDGIN8(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 8 - basic input \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return (_getAllowedSet(Chem.getType(atom), ELEM_LIST)
                +_getAllowedSet(Chem.getCIPConfiguration(atom), [8,2,4,0])
                +_getAllowedSet(Chem.getHybridizationState(atom), [1,2,3,7,8,1])
                +_getAllowedSet(atm_ring_length, [0,3,4,5,6,7,8,9,10,11]))

def atomFeaturesDGIN9(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 9 - basic input \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0]) 
                +list([float(Chem.isAmideCenterAtom(atom,mol))])
                +list([float(Chem.getAromaticityFlag(atom))])
                +_getAllowedSet(Chem.getImplicitHydrogenCount(atom), [0,1,2,3,4,5])))

def atomFeaturesDGIN10(atom,mol,atm_ring_length):
        ''' 
        generates the atom features of the DGIN 10 - basic input \n
        Input \n
        atom (CDPL BasicAtom): atom for the features calculation \n
        mol (CDPL BasicMolecule): molecule, the atom belongs to \n
        Return \n
        (list): binary atom feature list - 
        '''
        return ((_getAllowedSet(Chem.getType(atom), ELEM_LIST) # 11
                +_getAllowedSet(Chem.getFormalCharge(atom), [-1,-2,1,2,0])))


#########
# bond features

def bondFeaturesDMPNN(bond,mol,bond_ring_length=0):
    ''' 
    generates the edge features as in Yang et al. \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])
            +list([float(Chem.getAromaticityFlag(bond))]
            +list([float(_isConjugated(bond,mol))])
                +list([float(Chem.getRingFlag(bond))])
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))))

def bondFeaturesDGIN(bond,mol,bond_ring_length=0):
    ''' 
    generates the edge features of the DGIN 1 - e.g. as in model2_4_6. \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])
            +list([float(Chem.getAromaticityFlag(bond))]
            +list([float(_isConjugated(bond,mol))])
                +list([float(Chem.getRingFlag(bond))])
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))))

def bondFeaturesDGIN2(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 2 - approach with e.g. rot bond and others \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])
            +list([float(_isConjugated(bond,mol))])
                +list([float(Chem.isRotatable(bond,mol,False,True,True))])
                +list([float(Chem.isAmideBond(bond,mol))])
            +list([float(Chem.getAromaticityFlag(bond))]
                +list([float(Chem.getRingFlag(bond))])
                +_getAllowedSet(bond_ring_length, [0,3,4,5,6,7,8,9])
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))))

def bondFeaturesDGIN3(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 3 - approach with e.g. rot bond and others \n
    differs to DGIN2 in a different ring_length counter (1/ring_leng) instead of allowed set\n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3]))
            +list([float(_isConjugated(bond,mol))])
                +list([float(Chem.isRotatable(bond,mol,False,True,True))])
                +list([float(Chem.isAmideBond(bond,mol))])
            +list([float(Chem.getAromaticityFlag(bond))])
                +list([float(Chem.getRingFlag(bond))])
                +list([float(0.0) if bond_ring_length == float(0.0) else float(1/bond_ring_length)])
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))

def bondFeaturesDGIN4(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 4 - approach with e.g. rot bond and others \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])
            +list([float(_isConjugated(bond,mol))])
                +list([float(Chem.isRotatable(bond,mol,False,True,True))])
                +list([float(Chem.isAmideBond(bond,mol))])
            +list([float(Chem.getAromaticityFlag(bond))]
                +list([float(Chem.getRingFlag(bond))])
                +_getAllowedSet(bond_ring_length, [0,3,4,5,6,7,8,9,10,11])
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))))


def bondFeaturesDGIN5(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 5 - basic input \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])))

def bondFeaturesDGIN6(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 6 - approach with e.g. rot bond and others \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])
            +list([float(_isConjugated(bond,mol))])
                +list([float(Chem.isRotatable(bond,mol,False,True,True))])
                +list([float(Chem.isAmideBond(bond,mol))])
            +list([float(Chem.getAromaticityFlag(bond))]
                +list([float(Chem.getRingFlag(bond))])
                +_getAllowedSet(bond_ring_length, [0,3,4,5,6,7,8,9,10,11])
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))))

def bondFeaturesDGIN7(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 7 - basic input \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])))

def bondFeaturesDGIN8(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 8 - approach with e.g. rot bond and others \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return (_getAllowedSet(Chem.getOrder(bond), [1,2,3])
                +list([float(Chem.isRotatable(bond,mol,False,True,True))])
                +_getAllowedSet(bond_ring_length, [0,3,4,5,6,7,8,9,10,11]))

def bondFeaturesDGIN9(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 9 - approach with e.g. rot bond and others \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])
            +list([float(_isConjugated(bond,mol))])
                +list([float(Chem.isAmideBond(bond,mol))])
            +list([float(Chem.getAromaticityFlag(bond))]
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))))

def bondFeaturesDGIN10(bond,mol,bond_ring_length):
    ''' 
    generates the edge features of the DGIN 10 - approach with e.g. rot bond and others \n
    Input \n
    bond (CDPL BasicBond): bond that needs to be featurized \n
    mol (CDPL BasicMolecule): parent mol \n
    Return \n
    (list): binary edge feature list -
    '''
    return ((_getAllowedSet(Chem.getOrder(bond), [1,2,3])
                +list([float(Chem.isRotatable(bond,mol,False,True,True))])
            +list([float(Chem.getAromaticityFlag(bond))]
                +_getAllowedSet(bond_ring_length, [0,3,4,5,6,7,8,9,10,11])
            +_getAllowedSet(Chem.getCIPConfiguration(bond), [1,8,2,4,"Z","E"]))))    



# =============================================================================
# Read Write Methods
# =============================================================================


def readChemblXls(path_to_xls,col_entries = [0,7,10],sheet_index=0,n_entries=10000,skip_rows=0):
    '''
    reads a xls file and returns the name in column 1 with the corresponding smiles in column 2. \n
    Input: \n
    path_to_xls (string): path to the file.xls \n
    col_entries (list): the entries one wants to retrieve. Default: [0,7,10] \n
    sheet_index (int): Which sheet should be adressed. Default: 0 \n
    n_entries (int): How many rows are in the file and should be retieved. Default: 10000 \n
    skip_rows (int): Rows to skip from the beginning. DEFAULT=0 \n
    Returns: \n
    list: all values retrieved from the xls file
    
    '''
    wb = xlrd.open_workbook(path_to_xls)
    sheet = wb.sheet_by_index(sheet_index)
    nr_row = n_entries
    row_nr = 0
    data = list()
    names = list()
    try:
        for row in range(0, 1):
            for col_entry in col_entries:
                names.append(sheet.cell_value(row, col_entry))
        for row in range(skip_rows, nr_row):
            single_entry = []
            for col_entry in col_entries:
                single_entry.append(sheet.cell_value(row, col_entry))
            row_nr += 1
            single_entry.append(names)
            data.append(single_entry)
    except Exception as e:
        logging.info("End of xls file with",row_nr,"entries.",exc_info=True)
        pass
    return data

def CDPLmolFromSmiles(smiles_path,conformation,clean_structure=True):
    ''' 
    generates a CDPL Molecule from smiles. If confromations is true, then
    one random conformation will be generated with explicit hydrogens. \n
    Input: \n
    smiles_path (string): smiles_path to the smi file OR smiles string \n
    conformation (boolean): generates one 3d conformation according to MMFF94 \n
    clean_structure (boolean): cleans the structure by removing salts etc. (default = True) \n
    Return: \n
    (CDPL BasicMolecule): the corresponding CDPL BasicMolecule
    '''
    mol = Chem.BasicMolecule()
    if ".smi" in smiles_path:
        smi_reader = Chem.FileSMILESMoleculeReader(smiles_path)
        if not smi_reader.read(mol):
            logging.error("COULD NOT READ Smiles",smiles_path)
            return False
    else:
        mol = Chem.parseSMILES(smiles_path)
    if conformation:
        return _CDPLgenerateConformation(mol)
    if clean_structure:
        return _cleanMolecule(mol)
    return mol

def CDPLmolFromSdf(sdf_path,conformation):
    '''
    generates a single CDPL Molecule from an sdf-file. If conformations is true, then
    one random conformation will be generated. \n
    Input: \n
    sdf_path (string): path to the sdf file \n
    conformation (boolean): generates one 3d conformation according to MMFF94 \n
    Return: \n
    (CDPL BasicMolecule): the corresponding CDPL BasicMolecule 
    '''
    mol = Chem.BasicMolecule()
    ifs = Base.FileIOStream(sdf_path, 'r')
    sdf_reader = Chem.SDFMoleculeReader(ifs)

    if not sdf_reader.read(mol):
        logging.error("COULD NOT READ SDF",sdf_path)
        return False
    if conformation:
        return _CDPLgenerateConformation(mol)
    return mol

def pickleGraphs(path_to_folder,data,num_splits):
    ''' 
    pickles the input data list into the set folder and splits it according
    to the num_splits defined. \n
    Input: \n
    path_to_folder (string): path to the output folder \n
    data (list): list of graph instances \n
    num_splits (int): into how many chuncks the data should be split \n
    Return: \n
    (boolean): True, if the pickle worked, False otherwise
    '''
    try:
        if not os.path.isdir(path_to_folder):
            log.error("Not a valid path:"+path_to_folder,exc_info=True)
            return False
        le = (len(data) + num_splits - 1) / num_splits
        for split_id in range(num_splits):
            st = split_id * le
            sub_data = data[int(st) : int(st + le)]

            with open(path_to_folder+'graphs-%d.pkl' % split_id, 'wb') as f:
                pickle.dump(sub_data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        logging.error("saving issue",exc_info=True)
        return False

# =============================================================================
#  Miscellaneous Methods
# =============================================================================

def _getAllowedSet(x, allowable_set):
    ''' 
    PRIVATE METHOD 
    generates a one-hot encoded list for x. If x not in allowable_set,
    the last value of the allowable_set is taken. \n
    Input \n
        x (list): list of target values \n
        allowable_set (list): the allowed set \n
    Returns: \n
        (list): one-hot encoded list 
    '''
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: float(x == s), allowable_set))

def isValidFile(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
        return None
    else:
        return open(arg, 'r')  # return an open file handle

def make_batch(data,batch_size):
    '''
    create batches out of data with batch_size. The last batch is simply
    filled up with the last instance - Not nesessarily containing a batch_size amount.
    Input \n
        data (list): list of data \n
        batch_size (int): how many instances should be in the batch \n
    Returns: \n
        (list(batch_sized lists)): list of lists with batch_sized lists.
    '''
    batched_list = list()
    for i in range(0, len(data), batch_size):
        batched_list.append(data[i:i+batch_size])
    return batched_list


# =============================================================================
#  Miscellaneous Classes
# =============================================================================

class ColumnsAction(argparse.Action):
    def __call__(self, parser, namespace, value, option_string=None):
        self.validate(parser, value)
        setattr(namespace, self.dest, value)

    @staticmethod
    def validate(parser, value):
        if value not in ('foo', 'bar'):
            parser.error('{} not valid column'.format(value))

# =============================================================================
#  Neural Network relataed Methods/Classes
# =============================================================================

class CustomDropout(tf.keras.layers.Layer):

  def __init__(self, rate, **kwargs):
    super(CustomDropout, self).__init__(**kwargs)
    self.rate = rate

  def call(self, inputs, training=None):
    if training:
        return tf.nn.dropout(inputs, rate=self.rate)
    return inputs