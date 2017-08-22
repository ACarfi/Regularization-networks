# Regularization-networks
[![Licence](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/ACarfi/Regularization-networks/blob/master/LICENSE)

This project was built within the [MLCC 2017] and consist of the Python version of the [Lab II] MATLAB code.

### How to get started

The guide refers to Ubuntu 14.04 LTS, depending on the O.S. commands could differ.
In order to start using this project, navigate to the folder in which you want to work and clone the repository
```sh
$ git clone https://github.com/ACarfi/Regularization-networks.git
```
Check which python version you are running
```sh
$ python -V
```
The code was developed and tested using Python 2.7.6, it would not work for 3. Python versions.

### Project structure
The project is composed of two Python packages:
  - **regularizationNetworks**: contains all the functions used to implement and test regularization networks
  - **tests**: contains some usage example of the regularizationNetworks functions

### Code test
While located in the top folder **Regularization-networks** run
```sh
$ python -m tests.datasetCreation
```
This script will generate a 2D DataSet and asks the user with which name the DataSet should be saved. The DataSet is
saved with the provided name in the datasets folder, this allows the user to perform several test on the same data.

In the **KernelRegularizedLeastSquares.py** script substitute "insert_file_name.mat"
```sh
mat_contents = sio.loadmat('./datasets/insert_file_name.mat')
```
with the name you used to save your DataSet, then execute the script
```sh
$ python -m tests.KernelRegularizedLeastSquares
```
This scripts loads the saved data and displays it with two scatter plot, one for the training data and one for the test data.
Then the regularizationNetworks functions are used in order to compute and visualize the separating function in normal condition
and after inserting errors in the training data using the **flipLabel** function. These analyses are successively repeated for 
the **moon_dataset**.

The last example script loads the **moon_dataset** and used the **holdoutCVKernRLS** in order to find the value of **sigma** and
**lambda** that optimize the separating function, the resulting separating function is displayed as in the previous
script. In order to execute this script run 
```sh
$ python -m tests.datasetCreation
```

  [mlcc 2017]: <http://lcsl.mit.edu/courses/mlcc/mlcc2017/>
  [Lab II]: <http://lcsl.mit.edu/courses/master/MLCC/labs/lab2/index.html>
