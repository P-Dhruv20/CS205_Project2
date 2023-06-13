#ifndef PROJECT2_PROJECT_H
#define PROJECT2_PROJECT_H

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <list>
#include <algorithm>
#include <numeric>

using namespace std;

vector<vector<double>> dataset;

class Project {
private:
    int size;
public:
    int getSize() const {
        return this->size;
    }

    Project(int size) {
        this->size = size;
        srand(time(NULL));
    }

    static int nearest_neighbor(vector<double> a, vector<vector<double> > b) {
        double minDistance = 0.0;
        int classification = (int)a.at(0);
        for (int i = 0; i < b.at(0).size(); i++) {
            double distance = 0.0;
            for (int j = 1; j < b.size(); j++) { distance += pow(b.at(j).at(i) - a.at(j), 2); }
            if (i == 0) { minDistance = distance; }
            else if (distance < minDistance) {
                minDistance = distance;
                classification = (int) b.at(0).at(i);
            }
        }
        return classification;
    }

    static double leaveOneOutValidator(vector<vector<double> > currSet) {
        vector<vector<double>> instance_subset;
        vector<double> instanceCheck;
        int numCorrect = 0;
        for (int i = 0; i < currSet.at(0).size(); i++) {
            for (auto tmp: currSet) {
                instanceCheck.push_back(tmp.at(i));
                tmp.erase(tmp.begin() + i);
                instance_subset.push_back(tmp);
            }
            if ((int) nearest_neighbor(instanceCheck, instance_subset) == (int) instanceCheck.at(0)) { numCorrect++; }
            instance_subset.clear();
            instanceCheck.clear();
        }
        return ((double) numCorrect / (double) currSet.at(0).size()) * 100.00;
    }

    static void displayLocal(vector<int> localBest, double localmax) {
        cout << "The best feature set is { ";
        for (int i = 0; i < localBest.size(); i++) {
            cout << localBest.at(i);
            if (i < localBest.size() - 1) { cout << ", "; }
        }
        cout << " } with an accuracy of " << localmax << "%" << endl << endl;
    }

    static void displayBest(vector<int> bestFeatures, double max) {
        cout << endl << "Finished search!! The best feature set is { ";
        for (int i = 0; i < bestFeatures.size(); i++) {
            cout << bestFeatures.at(i);
            if (i < bestFeatures.size() - 1) { cout << ", "; }
        }
        cout << " }, with an accuracy of " << max << "%" << endl;
    }

    void forward_selection() {
        vector<int> bestFeatures;
        vector<int> localBest;
        double accuracy, max = 0.0;
        for (int i = 1; i <= getSize(); i++) {
            double localmax = 0.0;
            vector<int> tmpMax;
            for (int j = 1; j <= getSize(); j++) {
                vector<vector<double>> tmp;
                vector<int> tmpLocal;
                tmp.push_back(dataset.at(0));
                tmpLocal = localBest;

                for (int x: tmpLocal) { tmp.push_back(dataset.at(x)); }

                if (find(tmpLocal.begin(), tmpLocal.end(), j) == tmpLocal.end()) {
                    tmp.push_back(dataset.at(j));
                    tmpLocal.push_back(j);
                    cout << "Using feature(s) { ";
                    for (int i = 0; i < tmpLocal.size(); i++) {
                        cout << tmpLocal.at(i);
                        if (i < tmpLocal.size() - 1) { cout << ", "; }
                    }
                    accuracy = leaveOneOutValidator(tmp);
                    cout << " } accuracy is " << accuracy << "%" << endl;
                    if (accuracy > localmax) {
                        localmax = accuracy;
                        tmpMax = tmpLocal;
                    }
                }
            }
            localBest = tmpMax;
            displayLocal(localBest, localmax);
            if (localmax > max) {
                bestFeatures = localBest;
                max = localmax;
                localBest = bestFeatures;
            } else if (localmax < max) {
                cout << "Warning! Accuracy has decreased! Continuing search in case of local maxima." << endl;
                break;
            }
        }
        displayBest(bestFeatures, max);
    }

    void backward_elimination() {
        vector<int> bestFeatures(getSize());
        std::iota(bestFeatures.begin(), bestFeatures.end(), 1);
        vector<int> localBest = bestFeatures;
        double accuracy, max = 0.0;

        for (int i = 1; i < getSize(); i++) {
            double localmax = 0.0;
            vector<int> tmpMax;
            for (int j = 0; j <= getSize(); j++) {
                vector<vector<double>> tmp;
                vector<int> tmpLocal;
                tmp.push_back(dataset.at(0));
                tmpLocal = localBest;

                for (int x: localBest) {
                    tmp.push_back(dataset.at(x));
                }
                auto it = find(tmpLocal.begin(), tmpLocal.end(), j);
                if (it != tmpLocal.end()) {
                    int index = it - tmpLocal.begin() + 1;
                    tmpLocal.erase(it);
                    tmp.erase(tmp.begin() + index);
                    cout << "Using feature(s) { ";
                    for (int i = 0; i < tmpLocal.size(); i++) {
                        cout << tmpLocal.at(i);
                        if (i < tmpLocal.size() - 1) { cout << ", "; }
                    }
                    accuracy = leaveOneOutValidator(tmp);
                    cout << " } accuracy is " << accuracy << "%" << endl;
                    if (accuracy >= localmax) {
                        localmax = accuracy;
                        tmpMax = tmpLocal;
                    }
                }
            }
            localBest = tmpMax;
            displayLocal(localBest, localmax);
            if (localmax > max) {
                bestFeatures = localBest;
                max = localmax;
            } else if (localmax < max) {
                cout << "Warning! Accuracy has decreased! Continuing search in case of local maxima." << endl;
                break;
            }
        }
        displayBest(bestFeatures, max);
    }

    static void normalizeData() {
        cout << endl << "Please wait while I normalize the data ... ";
        vector<pair<double, double>> normData;
        for (auto &i: dataset) {
            double min = INFINITY;
            double max = -1.0 * INFINITY;
            for (double j: i) {
                if (j < min) min = j;
                if (j > max) max = j;
            }
            normData.emplace_back(min, max);
        }
        for (int i = 0; i < dataset.size(); i++) {
            for (int j = 0; j < dataset.at(i).size(); j++) {
                dataset.at(i).at(j) =
                        (dataset.at(i).at(j) - normData.at(i).first) / (normData.at(i).second - normData.at(i).first);
            }
        }
        cout << "Done!" << endl << endl;
    }

    static double defaultRate() {
        double accuracy;
        int numClass1 = 0;
        int numClass2 = 0;

        for (double i: dataset.at(0)) {
            if (i == 1) numClass1++;
            else numClass2++;
        }
        if (numClass1 > numClass2) accuracy = 100.0 * (double) numClass1 / dataset.at(0).size();
        else accuracy = 100.0 * (double) numClass2 / dataset.at(0).size();
        return accuracy;
    }

    void search(int choice) {
        cout << "This dataset has " << getSize() << " features (not including the class attribute), with "
             << dataset.at(0).size() << " instances." << endl;
        normalizeData();
        if (choice == 1) {
            cout
                    << "Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of "
                    << defaultRate() << "%" << endl
                    << endl;
            cout << "Beginning search." << endl << endl;
            forward_selection();
        } else if (choice == 2) {
            cout
                    << "Running nearest neighbor with ALL features, using \"leaving-one-out\" evaluation, I get an accuracy of "
                    << leaveOneOutValidator(dataset) << "%" << endl
                    << endl;
            cout << "Beginning search." << endl << endl;
            backward_elimination();
        } else {
            cout << "Error! Not a correct selection from options" << endl;
        }
    }
};

#endif