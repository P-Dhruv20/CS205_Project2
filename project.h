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

    // Expects data in row-major order
    static int nearest_neighbor(const int queryIdx, const vector<int>& features) {
        double minDistance = 0.0;
        const vector<double>& queryInstance = dataset.at(queryIdx);
        int classification = static_cast<int>(dataset.at(queryIdx).at(0));
        const int numInstances = dataset.size();

        for (int rowIdx = 0; rowIdx < numInstances; rowIdx++) {
            if (rowIdx == queryIdx)
                continue;

            double distance = 0.0;
            for (int featureIdx : features) {
                 const double featureDist = dataset.at(rowIdx).at(featureIdx) - queryInstance.at(featureIdx);
                 distance += featureDist * featureDist;
            }

            if (rowIdx == 0) {
                 minDistance = distance;
            }
            else if (distance < minDistance) {
                minDistance = distance;
                classification = static_cast<int>(dataset.at(rowIdx).at(0));
            }
        }

        return classification;
    }

    // Performs leave one out validation (w/ k = 1)
    // Returns the number of correct predictions.
    static int leaveOneOutValidator(const vector<int>& features, int wrongLimit) {
        const int numInstances = dataset.size();

        int numWrong = 0;

        for (int leftOutRowIdx = 0; leftOutRowIdx < numInstances; leftOutRowIdx++) {
            const int predictedClass = nearest_neighbor(leftOutRowIdx, features);
            const int expectedClass = static_cast<int>(dataset.at(leftOutRowIdx).at(0));

            if (predictedClass != expectedClass) {
                 numWrong++;
                 // Implement early abandoning
                 // If number of wrong 
                 if (numWrong > wrongLimit) {
                    break;
                 }
            }
        }

        return numInstances - numWrong;
    }

    static void displayLocal(const vector<int>& localBest, double localmax) {
        cout << "The best feature set is { ";
        for (int i = 0; i < localBest.size(); i++) {
            cout << localBest.at(i);
            if (i < localBest.size() - 1) { cout << ", "; }
        }
        cout << " } with an accuracy of " << localmax << "%" << endl << endl;
    }

    static void displayBest(const vector<int>& bestFeatures, double max) {
        cout << endl << "Finished search!! The best feature set is { ";
        for (int i = 0; i < bestFeatures.size(); i++) {
            cout << bestFeatures.at(i);
            if (i < bestFeatures.size() - 1) { cout << ", "; }
        }
        cout << " }, with an accuracy of " << max << "%" << endl;
    }

    void forward_selection() {
        const int numFeatures = dataset.at(0).size();
        const int numInstances = dataset.size();
        vector<int> bestFeatures;

        int maxCorrect = 0;
        for (int depth = 1; depth < numFeatures; ++depth) {
            cout << "Exploring accuracy of NN with " << depth << " features!" << endl;

            // Current max correctly predicted instances at feature selection depth.
            int depthMaxCorrect = 0;
            // The current set of best features at this depth.
            vector<int> depthBestFeatures;

            // Early abandon leave-one-out tests that 
            // exceed the wrong prediction threshold.
            const int wrongLimit = numInstances - maxCorrect;

            for (int featureNum = 1; featureNum < numFeatures; ++featureNum) {
                // If the feature is not in the best feature list, then explore it
                if (find(bestFeatures.begin(), bestFeatures.end(), featureNum) == bestFeatures.end()) {
                    vector<int> currFeatures = bestFeatures;
                    currFeatures.push_back(featureNum);

                    cout << "Using feature(s) { ";
                    for (int featureIdx : bestFeatures) {
                        cout << featureIdx << ", ";
                    }
                    cout << featureNum;

                    const int numCorrect = leaveOneOutValidator(currFeatures, wrongLimit);

                    const double accuracy = (100.0 * (numCorrect / static_cast<double>(numInstances - 1)));
                    cout << " } accuracy is " << accuracy << "%";

                    if (numCorrect <= maxCorrect) {
                        cout << " (Early Abandoned)";
                    }

                    cout << endl;

                    if (numCorrect > depthMaxCorrect) {
                        depthMaxCorrect = numCorrect;
       
                        depthBestFeatures = bestFeatures;
                        depthBestFeatures.push_back(featureNum);
                    }
                }
            }

            displayLocal(depthBestFeatures, (100.0 * (depthMaxCorrect / static_cast<double>(numInstances - 1))));
            if (depthMaxCorrect > maxCorrect) {
                bestFeatures = depthBestFeatures;
                maxCorrect = depthMaxCorrect;
            } else if (depthMaxCorrect < maxCorrect) {
                cout << "Warning! Accuracy has decreased! Continuing search in case of local maxima." << endl;
                break;
            }
        }

        displayBest(bestFeatures, (100.0 * (maxCorrect / static_cast<double>(numInstances))));
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
                    accuracy = leaveOneOutValidator({ 1, 2, 3 }, 1000); // TODO FIX
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

    // Normalize the data via min-max normalization on each feature.
    static void normalizeData() {
        cout << endl << "Please wait while I normalize the data ... ";\
        const int numFeatures = dataset.at(0).size();
        // Vector of the minimum and maximum for each feature
        vector<pair<double, double>> featureMinMax(numFeatures - 1, { INFINITY, -1 * INFINITY });

        // Go through dataset and find actual min & max for each feature
        for (const vector<double>& row : dataset) {
            for (int featureIdx = 1; featureIdx < numFeatures; ++featureIdx) {
                const double featureVal = row.at(featureIdx);
                auto& [min, max] = featureMinMax.at(featureIdx - 1);
                
                if (featureVal < min) min = featureVal;
                if (featureVal > max) max = featureVal;
            }
        }

        // Go through dataset again and apply min max normalization
        for (vector<double>& row : dataset) {
            for (int featureIdx = 1; featureIdx < numFeatures; ++featureIdx) {
                double& featureVal = row.at(featureIdx);
                const auto& [min, max] = featureMinMax.at(featureIdx - 1);
                featureVal = (featureVal - min) / (max - min);
            }
        }

        cout << "Done!" << endl << endl;
    }

    static double defaultRate() {
        int numClass1 = 0, numClass2 = 0;

        for (const vector<double>& row : dataset) {
            const double class_label = row.at(0);
            if (class_label == 1)
                numClass1++;
            else
                numClass2++;
        }

        int biggestClass = max(numClass1, numClass2);
        double accuracy = 100.0 * (static_cast<double>(biggestClass) / dataset.size());

        return accuracy;
    }

    void search(int choice) {
        const int numRows = dataset.size();
        const int numFeatures = dataset.at(0).size();

        cout << "This dataset has " << (numFeatures - 1) << " features (not including the class attribute), with "
             << numRows << " instances." << endl;

        normalizeData();

        if (choice == 1) {
            cout
                    << "Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of "
                    << defaultRate() << "%" << endl
                    << endl;
            cout << "Beginning search." << endl << endl;
            forward_selection();
        } else if (choice == 2) {
            std::vector<int> allFeatures(numFeatures - 1);
            std::iota(allFeatures.begin(), allFeatures.end(), 1);

            cout
                    << "Running nearest neighbor with ALL features, using \"leaving-one-out\" evaluation, I get an accuracy of "
                    << leaveOneOutValidator(allFeatures, numRows) << "%" << endl
                    << endl;
            cout << "Beginning search." << endl << endl;
            backward_elimination();
        } else {
            cout << "Error! Not a correct selection from options" << endl;
        }
    }
};

#endif