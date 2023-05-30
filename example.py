from sgdataset import SizeGroupedDataset
import numpy as np

ds = SizeGroupedDataset()

reactant_coord = np.array([[-0.005248003463, -0.000000000049, 0.000000000005],
                      [1.159999999836, 0.000000000098, -0.000000000010],
                      [2.325248003627, -0.000000000049, 0.000000000005]])
reactant_numbers = np.array([8,6,8])

product_coord = np.array([[-5.005248003463, -0.000000000049, 0.000000000005],
                      [-4.159999999836, 0.000000000098, -0.000000000010],
                      [-3.325248003627, -0.000000000049, 0.000000000005]])
product_numbers = np.array([8,6,8])
product_net_charge = np.array([[0]])

rxn_net_charge = np.array([[0]])

''' 
RESHAPING 
    THE SIZE GROUPED DATASET IS DESIGNED TO GROUP SYSTEM BY THE
    NUMBER OF ATOMS TO MAKE BATCHING AND MLIP TRAINING SIMPLE.
    WHEN WORKING WITH CHEMICAL REACTIONS, WE STACK THE REACTANTS
    AND PRODUCTS INTO A SINGLE SYSTEM, AND THEY ARE SEPARATED 
    LATER AFTER WE APPLY OUR QM AND ML SAMPLING WORKFLOWS.
'''

coord = np.vstack([reactant_coord,product_coord])
numbers = np.vstack([reactant_numbers, product_numbers])
charge = rxn_net_charge

''' coordinates '''
coord = coord.reshape(1,-1,3) # SIZE IS (BATCH,N_ATOMS*2,3)
''' atomic numbers '''
numbers = numbers.reshape(1,-1) # SIZE IS (BATCH,N_ATOMS*2)
''' net charge '''
charge = charge.flatten() # SIZE IS (BATCH)
''' name for bookkeeping, should match your dataset '''
_id = np.array(['rxn_number_1723742'])[()].astype('S') # THIS IS SIZE (BATCH)


d = dict(_id=_id, coord=coord, numbers=numbers, charge=charge)
dd = dict()
dd[coord.shape[-2]] = d # NOTE: THE SGD KEY IS THE NUMBER OF ATOMS IN THE SYSTEM
dd = SizeGroupedDataset(dd)

''' 
YOU CAN COMBINE SIZE GROUPED DATASETS THAT CONTAIN SYSTEMS OF
DIFFERENT SIZES USING THE MERGE FUNCTION. FOR EXAMPLE, IF YOU
HAVE 2 SGDs (DD1 AND DD2) YOU CAN ADD DD2 INTO DD1 BY DOING.

DD1.MERGE(DD2)

AFTER YOU HAVE GONE THROUGH AND USED MERGE TO CREATE A SINGLE 
SGD CONTAINING ALL THE SYSTEMS YOU WANT TO SEND, THE SGD CAN
BE SAVED AS A FORMATED H5 FILE USING THE FOLLOWING COMMAND
'''

dd.save_h5('this_is_my_dataset.h5')
del(dd)

'''
YOU CAN CHECK TO MAKE SURE THE H5 FILE WAS CREATED PROPERLY BY
PRINTING THE SHAPES USING THE FOLLOWING COMMANDS
'''

ds_check = SizeGroupedDataset('this_is_my_dataset.h5')
for k,v in ds_check.items():
    print(k, v['_id'].shape, v['coord'].shape, v['numbers'].shape, v['charge'].shape)


'''
FOR THE EXAMPLE, THIS PRINTS: 6 (1,) (1, 6, 3) (1, 6) (1,)
INDICATING THAT WE HAVE A 6 ATOM SYSTEM WITH A BATCH SIZE OF 1
'''
