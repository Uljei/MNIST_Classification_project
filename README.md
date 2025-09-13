Architecture:

MINIST_Classidication_project
|___ __init__.py
|___ model.py # backbone that fllows the project requirements
|___ Functions.py # store all the functions
|___ Visualization.py # for plots
|___ HyperparamStudy.py # run indivisually to achieve hyperparameter study
|___ tests/
     |___  __init__.py
     |___ test_activations_and_losses.py
     |___ test_attack.py
     |___ test_intergration.py
     |___ test_layers.py
     |___ test_model.py
     |___ test_visualization.py

One can implement the project by following the steps below.
<1> Run Functions.py and Visualization.py first for initiualizatoin
<2> Run model.py gives all the results as required in the project. 
    You can adjust the model parameters, inclusing the activation fuctions, loss, layers, and number of nerouns, as needed.
<3> Run HyperparamStudy.py for detailed hyperparameter study.
<4> Test with the command 
    '''
    bash
    PYTHONPATH=. pytest tests/ -v
    '''

