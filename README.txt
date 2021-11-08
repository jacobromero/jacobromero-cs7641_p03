### To run this code you need
* Python 3.x

### and the following packages
* numpy
* pandas
* matplotlib
* scikitlearn

### Dataset urls
* [Dataset 1](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset)
* [Dataset 2](https://archive.ics.uci.edu/ml/datasets/adult)

### Pulling the code, 
The code is hosted on github at https://github.com/jacobromero/cs7641_p01 to get this code simply do:
`git clone https://github.com/jacobromero/cs7641_p03.git`

Alternatively you can download the code from google drive at, [Redacted]
### To Run this code
The project is intended to be run as a python juypter note book via `jupyter notebook` command.
This requires juypter installed via `pip install juypter`.

Alternatively the notebook has been exported to python from the juypter notebook, and can be run with
`python3 project_03_notebook.py`
However it is recommended to view the code via the notebook still.

#### Loading the data
The code has been organized so that the code (python script/notebook) will load the data from the same directory the code is ran in.
Specifically from a folder titled 'data', if the data fails to load ensure that the path to the data files are:
 - `data/adult.data`, and `data/adult.test`
 - 'data/UCI_Credit_Card.csv'