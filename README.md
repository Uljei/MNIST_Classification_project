Architecture:
MINIST_Classidication_project
|___ __init__.py
|___ model_demo.ipynb # jupyter notebook for presentation
|___ model.py # backbone that follows the project requirements
|___ Functions.py # store all the functions
|___ Visualization.py # for plots
|___ HyperparamStudy.py # run individually for hyperparameter study
|___ tests/
     |___  __init__.py
     |___ test_activations_and_losses.py
     |___ test_attack.py
     |___ test_intergration.py
     |___ test_layers.py
     |___ test_model.py
     |___ test_visualization.py
How to use?
We provide a notbook file model_demo.ipynb for a clear presentation.
One can also implement the project by following the steps:
<1> Run Functions.py and Visualization.py first for initializatoin
<2> Run model.py gives all the results as required in the project.
    You can adjust the model parameters, including the activation functions, loss, layers, and number of nerouns, as needed.
<3> Run HyperparamStudy.py for detailed hyperparameter study.
<4> Test with the command
    '''
    bash
    cd YOUR/PATH/TO/MINIST_Classidication_project
    PYTHONPATH=. pytest tests/ -v
    '''
Group contributions:
Zhe Zhang:
Jiazhuang Chen:
Ruizhen Shen: Implemented the FGSM attack function and applied it to the trained model.
Jiuen Feng: Train again with additional FGSM attack data. Implemented the DeepFool attack.
Jia Gu: Organized .py files into modules; tested the code and fixed any bugs found.
