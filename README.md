# CS205_Project2

**Authors: Dhruv Parmar, Sanchit Goel**

Disclaimers: 
* We are not the owners or creators of the datasets used in this project.
* This project is for educational purposes only and should not be plagerized for educational tasks.

This repository hosts the datasets, code, and outputs for a feature selection program utilizing nearest neighbor classification.
The two feature selections methods used are forward selection and backwards elimination.

## Usage

The project creates two executables: `CS205_Project2` and `CleanupBCData`. The main executable is the `CS205_Project2` which prompts the user for a dataset and the feature selection to use. The program then performs and outputs a trace of the feature selection search utilized.

The datasets used must follow a specific format:
* All feature values must be non-empty in numerical format.
* Class labels must be placed in the first column.
* Features must be space-delimited.
* The data file must not have any headers.

The `CleanupBCData` cleans up the Wisconsin Breast Cancer dataset to have it match the format specified above. It expects the dataset to be in a dataset directory accessible by the executable (specifically with the relative path `../data/breast-cancer.csv`). The cleaned dataset is placed in the relative path: `../data/processed_breast-cancer.csv`.

## Building
This project is a very simple CMake project.

Requirements:
* CMake 3.25 or above.
* C++ 17

1. Clone the project repository.
2. Inside of the project directory, create a new build directory (e.g. called `build`). 
3. Inside of the build directory, initialize CMake by running the command `cmake ..`
4. Create the executables by running the command `make` inside of the build directory.

Both executables should have been created inside of the build directory.


