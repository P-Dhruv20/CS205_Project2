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

    static int nearest_neighbor(const vector<double>& queryInstance, const vector<vector<double>>& data) {
        double minDistance = 0.0;
        int classification = static_cast<int>(queryInstance.at(0));
        const int numInstances = data.at(0).size();

        for (int rowIdx = 0; rowIdx < numInstances; rowIdx++) {
            double distance = 0.0;
            for (int featureIdx = 1; featureIdx < data.size(); featureIdx++) {
                 const double featureDist = data.at(featureIdx).at(rowIdx) - queryInstance.at(featureIdx);
                 distance += featureDist * featureDist;
            }

            if (rowIdx == 0) {
                 minDistance = distance;
            }
            else if (distance < minDistance) {
                minDistance = distance;
                classification = static_cast<int>(data.at(0).at(rowIdx));
            }
        }

        return classification;
    }

    // Performs leave one out validation (w/ k = 1)
    // Returns the number of correct predictions.
    static int leaveOneOutValidator(const vector<vector<double>>& currData, int wrongLimit) {
        const int numInstances = currData.at(0).size();

        vector<vector<double>> instance_subset;
        vector<double> testInstance;
        int numWrong = 0;

        for (int leftOutRowIdx = 0; leftOutRowIdx < numInstances; leftOutRowIdx++) {
            for (const auto& column : currData) {
                testInstance.push_back(column.at(leftOutRowIdx));

                // Add remaining instances in the colunmn
                vector<double> columnSubset;
                columnSubset.reserve(column.size() - 1);
                for (int row = 0; row < column.size(); row++) {
                    if (row == leftOutRowIdx)
                        continue;
                    
                    columnSubset.push_back(column.at(row));
                }
                
                instance_subset.emplace_back(std::move(columnSubset));
            }

            const int predictedClass = nearest_neighbor(testInstance, instance_subset);
            const int expectedClass = static_cast<int>(testInstance.at(0));

            if (predictedClass != expectedClass) {
                 numWrong++;
                 if (numWrong > wrongLimit) {
                    break;
                 }
            }

            instance_subset.clear();
            testInstance.clear();
        }

        return numInstances - numWrong;
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
        const int numFeatures = dataset.size();
        const int numInstances = dataset.at(0).size();
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
                vector<vector<double>> reducedData;
                reducedData.reserve(bestFeatures.size() + 2);

                reducedData.push_back(dataset.at(0));

                for (int featureIdx : bestFeatures) {
                    reducedData.push_back(dataset.at(featureIdx));
                }

                // If the feature is not in the best feature list, then explore it
                if (find(bestFeatures.begin(), bestFeatures.end(), featureNum) == bestFeatures.end()) {
                    reducedData.push_back(dataset.at(featureNum));

                    cout << "Using feature(s) { ";
                    for (int featureIdx : bestFeatures) {
                        cout << featureIdx << ", ";
                    }
                    cout << featureNum;

                    const int numCorrect = leaveOneOutValidator(reducedData, wrongLimit);

                    const double accuracy = (100.0 * (numCorrect / static_cast<double>(numInstances)));
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

            displayLocal(depthBestFeatures, (100.0 * (depthMaxCorrect / static_cast<double>(numInstances))));
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
                    accuracy = leaveOneOutValidator(tmp, 1000); // TODO FIX
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
        cout << endl << "Please wait while I normalize the data ... ";
        for (auto& features: dataset) {
            // Compute the min and max on the feature
            double min = INFINITY;
            double max = -1.0 * INFINITY;
            for (double instance : features) {
                if (instance < min) min = instance;
                if (instance > max) max = instance;
            }
            // Apply normalization to the feature
            double denom = max - min;
            for (double& instance : features) {
                instance = (instance - min) / denom;
            }
        }

        cout << "Done!" << endl << endl;
    }

    static double defaultRate() {
        int numClass1, numClass2;
        
        vector<double>& class_labels = dataset.at(0);

        for (const double class_label : class_labels) {
            if (class_label == 1)
                numClass1++;
            else
                numClass2++;
        }

        int biggestClass = max(numClass1, numClass2);
        double accuracy = 100.0 * (static_cast<double>(biggestClass) / class_labels.size());

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
                    << leaveOneOutValidator(dataset, static_cast<int>(dataset.at(0).size())) << "%" << endl
                    << endl;
            cout << "Beginning search." << endl << endl;
            backward_elimination();
        } else {
            cout << "Error! Not a correct selection from options" << endl;
        }
    }
};

#endif