#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include "project.h"

using namespace std;

int main() {
    int numFeat, choice = 0;
    string filename, line, feature;
    ifstream file;
    vector<double> instance;
    vector<double> features;
    cout << "Welcome to Dhruv Parmar's Feature Selection Algorithm." << endl;
    cout << endl;
    cout << "Type in the name of the file to test: " << endl;
    cin >> filename;
    cout << endl << "Type the number of the algorithm you want to run." << endl;
    cout << "1. Forward Selection" << endl;
    cout << "2. Backward Elimination" << endl;
    cin >> choice;
    cout << endl;
    ifstream fileRead;
    filename = "../" + filename;
    fileRead.open(filename.c_str());
    if (!fileRead.is_open()) {
        cout << "Error opening file." << endl;
        return -1;
    }
    getline(fileRead, line);
    std::stringstream sstream(line);
    while (sstream >> feature) {
        features.push_back(stold(feature));
        dataset.push_back(features);
        features.clear();
    }
    while (getline(fileRead, line)) {
        std::stringstream sstream(line);
        string feature;
        for (auto &i: dataset) {
            if (sstream >> feature) { i.push_back(stold(feature)); }
        }
    }
    fileRead.close();
    numFeat = dataset.size() - 1;
    Project project = Project(numFeat);
    project.search(choice);
    return 0;
}