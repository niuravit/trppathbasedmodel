# trppathbasedmodel

1) Open generatinginstance notebook and run all blocks to generate some random instances (save to folder '/instances')
2) Open solvingtrp notebook and run the file to solve the trp path based model

*Path to the Gurobi license file can be set in the file: 'modules/model.py'*
List of required lib:
- numpy
- random
- pandas
- pickle
- gurobipy
- plotly (for plotting: https://anaconda.org/conda-forge/plotly)
- nltk (for generating bigram, used for arc generation of the network: https://anaconda.org/anaconda/nltk)
