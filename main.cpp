#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "project.h"

using namespace std;

// Retrieves the dataset column-major wise.
// I.e. the outer vector refers to the features, the inner vector refers to the rows.
vector<vector<double>> read_data(ifstream& file_stream) {
    vector<vector<double>> dataset; 

    // Preload the dataset with the feature vectors
    string line;
    {
        string feature;
        getline(file_stream, line);
        std::stringstream sstream(line);
        while (sstream >> feature) {
            dataset.push_back(vector<double>{ static_cast<double>(stold(feature)) });
        }
    }

    // Put instances into the dataset vector
    while (getline(file_stream, line)) {
        std::stringstream sstream(line);
        for (auto& column: dataset) {
            string column_val;
            if (sstream >> column_val) {
                 column.push_back(stold(column_val));
            }
        }
    }

    return dataset;
}

int main() {
    int numFeat, choice = 0;
    string filename;
    ifstream file;

    cout << "Welcome to Dhruv Parmar's Feature Selection Algorithm." << endl;
    cout << endl;
    cout << "Type in the name of the dataset to test: " << endl;
    cin >> filename;
    cout << endl << "Type the number of the algorithm you want to run." << endl;
    cout << "1. Forward Selection" << endl;
    cout << "2. Backward Elimination" << endl;
    cin >> choice;
    cout << endl;

    ifstream fileRead;
    fileRead.open(filename.c_str());
    if (!fileRead.is_open()) {
        cout << "Error opening file." << endl;
        return -1;
    }

    dataset = read_data(fileRead);
    fileRead.close();

    numFeat = dataset.size() - 1;
    Project project = Project(numFeat);
    project.search(choice);
    return 0;
}