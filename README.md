# Regularization-networks

This project was built within the [MLCC 2017] and consist of the Python version of the [Lab II] MATLAB code.

### How to get started

The guide refers to Ubuntu 14.04 LTS, depending on the O.S. commands could difer.
In order to start using this project, navigate to the folder in which you want to work and clone the repository
```sh
$ git clone https://github.com/ACarfi/Regularization-networks.git
```
Check which python version you are running
```sh
$ python -V
```
The code was developed and tested using Python 2.7.6, it should work fine with different Python versions but keep in mind tha this is a possible source of errors.

### Project structure
The project is composed of two Python packages:
  - **regularizationNetworks**: contains all the functions used to implement and test regularization networks
  - **tests**: contains some usage example of the regularizationNetworks functions

### Code test
In the top folder **Regularization-networks** run
```sh
$ python -m tests.datasetCreation
```
This script will generate a 2D DataSet and asks the user with which name the DataSet should be saved. The DataSet is
saved with the provided name in the datasets folder, this allows the user to perform several test on the same DataSet.


  [mlcc 2017]: <http://lcsl.mit.edu/courses/mlcc/mlcc2017/>
  [Lab II]: <http://lcsl.mit.edu/courses/master/MLCC/labs/lab2/index.html>