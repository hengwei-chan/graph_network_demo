graph_networks
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/graph_networks/branch/master)


prediction of chemical properties with graphs and neural networks


### Basic usage
# Generate graph instances
Use an xls file as an input - required arguments are:
1. --input_file_path    -set the input files
2. --output_path    -set the output folder (the graph instances are being pickled)
3. --featurization  -set what kind of featurization is needed
4. --columns    -select the columns to be read in

~ python ./graph_networks/scripts/generate_graphs.py --input_file_path ./data/lipo_plus.xls --output_path ./graph_networks/data/output_folder/ --columns 0 1 2 3 4
--featurization DGIN3

# define config file


# Train GNN with graph instances

### Copyright

Copyright (c) 2021, oliver wieder


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.5.
