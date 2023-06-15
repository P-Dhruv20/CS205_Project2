#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "project.h"

using namespace std;

// Retrieves the dataset row-major wise.
// I.e. the outer vector refers to the rows, the inner vector refers to the features.
vector<vector<double>> read_data(ifstream& file_stream) {
    vector<vector<double>> dataset; 

    int numFeatures = -1;

    // Put instances into the dataset vector
    string line;
    while (getline(file_stream, line)) {
        std::stringstream sstream(line);
        string feature;
        vector<double> row;
        if (numFeatures > 0)
            row.reserve(numFeatures);

        while (sstream >> feature) {
            row.push_back(static_cast<double>(stold(feature)));
        }
        dataset.push_back(row);

        if (numFeatures < 0)
            numFeatures = row.size();
    }

    return dataset;
}

int main() {
    int numFeat, choice = 0;

    string filename;
    ifstream file;

    cout << "Welcome to Feature Selection Algorithm." << endl;
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

    numFeat = dataset.at(0).size() - 1;
    Project project = Project(numFeat);
    project.search(choice);
    return 0;
}