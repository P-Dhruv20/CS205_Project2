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

    // Performs leave-one-out validation but moves the instances
    // that were predicted wrong to the top
    static void reorderDataset(const vector<int>& features) {
        const int numInstances = dataset.size();

        vector<int> mispredictedInstances;

        for (int leftOutRowIdx = 0; leftOutRowIdx < numInstances; leftOutRowIdx++) {
            const int predictedClass = nearest_neighbor(leftOutRowIdx, features);
            const int expectedClass = static_cast<int>(dataset.at(leftOutRowIdx).at(0));

            if (predictedClass != expectedClass) {
                mispredictedInstances.push_back(leftOutRowIdx);
            }
        }

        // Swapping works because the mispredicted indexes
        // are guranteed to be in order.
        int swapIdx = 0;
        for (int mispredictedIdx : mispredictedInstances) {
            vector<double> temp = std::move(dataset.at(swapIdx));
            dataset.at(swapIdx) = std::move(dataset.at(mispredictedIdx));
            dataset.at(mispredictedIdx) = std::move(temp);
        }
    }

    static void displayLocal(const vector<int>& localBest, double localmax) {
        cout << "The feature set best at this depth is { ";
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

            if (depth > 1) {
                reorderDataset(bestFeatures);
                cout << "Re-ordered dataset for quicker early abandoning!" << endl;
            }

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
                    cout << featureNum << " } ";

                    const int numCorrect = leaveOneOutValidator(currFeatures, wrongLimit);

                    const double accuracy = (100.0 * (numCorrect / static_cast<double>(numInstances - 1)));
                    if (numCorrect <= maxCorrect) {
                        cout << "was early abandoned.";
                    }
                    else {
                        cout << "accuracy is " << accuracy << "%";
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
        const int numInstances = dataset.size();
        const int numFeatures = dataset.at(0).size();

        vector<int> bestFeatures(numFeatures - 1);
        std::iota(bestFeatures.begin(), bestFeatures.end(), 1);

        int maxCorrect = 0;

        for (int depth = 1; depth < numFeatures; ++depth) {
            int depthMaxCorrect = 0;
            vector<int> depthBestFeatures;

            if (depth > 1) {
                reorderDataset(bestFeatures);
                cout << "Re-ordered dataset for quicker early abandoning!" << endl;
            }

            const int wrongLimit = numInstances - maxCorrect;
            for (const int rmFeatureNum : bestFeatures) {
                vector<int> currFeatures;
                currFeatures.reserve(bestFeatures.size() - 1);
                cout << "Using feature(s) {";
                for (int featureNum : bestFeatures) {
                    if (rmFeatureNum ==  featureNum) {
                        continue;
                    }

                    cout << " " << featureNum;
                    currFeatures.push_back(featureNum);
                }
                cout << " } ";

                const int numCorrect = leaveOneOutValidator(currFeatures, wrongLimit);
                const double accuracy = (100.0 * (numCorrect / static_cast<double>(numInstances - 1)));
                if (numCorrect <= maxCorrect) {
                    cout << "was early abandoned.";
                }
                else {
                    cout << "accuracy is " << accuracy << "%";
                }

                cout << endl;

                if (numCorrect > depthMaxCorrect) {
                    depthMaxCorrect = numCorrect;
                    depthBestFeatures = currFeatures;
                }
            }

            const double depthBestAccuracy = (100.0 * (depthMaxCorrect / static_cast<double>(numInstances - 1)));
            displayLocal(depthBestFeatures, depthBestAccuracy);
            if (depthMaxCorrect > maxCorrect) {
                bestFeatures = depthBestFeatures;
                maxCorrect = depthMaxCorrect;
            } else if (depthMaxCorrect < maxCorrect) {
                cout << "Warning! Accuracy has decreased! Continuing search in case of local maxima." << endl;
                break;
            }
        }

        const double bestAccuracy = (100.0 * (maxCorrect / static_cast<double>(numInstances - 1)));
        displayBest(bestFeatures, bestAccuracy);
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