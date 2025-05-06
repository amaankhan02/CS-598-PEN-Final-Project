# CS 598 PEN FINAL PROJECT

# Environment Setup:
Use python version 3.11.11
Pip: 
```
pip install -r requirements.txt
```
Conda: 
```
conda env create -f environment.yml
```

Important libraries to install:
```
ray[rllib]==2.37
torch
numpy
google-generativeai
```

# Build Instructions
To run the trainer, first update the `config.py` file with any of your specific configurations. Then run 
```
python train.py <num_iterations>
```
`num_iterations` is an optional argument. If left blank, it will run the default number of iterations described in the `config.py` file
