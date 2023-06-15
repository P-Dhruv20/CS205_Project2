#include "project.h"

#include <iostream>
#include <string>
#include <cmath>
#include <list>
#include <algorithm>
#include <numeric>

Project::Project(vector<vector<double>> dataset)
    : dataset(std::move(dataset))
{
}

// find class of the nearest neighbor using Euclidean distance
int Project::nearest_neighbor(const int queryIdx, const vector<int> &features)
{
    double minDistance = 0.0;
    const vector<double> &queryInstance = dataset.at(queryIdx);
    int classification = static_cast<int>(dataset.at(queryIdx).at(0));
    const int numInstances = dataset.size();
    // loop through all instances except the query instance itself
    for (int rowIdx = 0; rowIdx < numInstances; rowIdx++)
    {
        if (rowIdx == queryIdx)
            continue;
        // find Euclidean distance between query instance and current instance
        double distance = 0.0;
        for (int featureIdx : features)
        {
            const double featureDist = dataset.at(rowIdx).at(featureIdx) - queryInstance.at(featureIdx);
            distance += featureDist * featureDist;
        }
        // if the distance is smaller than the current minimum distance, update the minimum distance
        if (rowIdx == 0)
        {
            minDistance = distance;
        }
        else if (distance < minDistance)
        {
            minDistance = distance;
            classification = static_cast<int>(dataset.at(rowIdx).at(0));
        }
    }

    return classification;
}

// Performs leave one out validation (w/ k = 1)
// Returns the number of correct predictions.
int Project::leaveOneOutValidator(const vector<int> &features, int wrongLimit)
{
    const int numInstances = dataset.size();

    int numWrong = 0;

    for (int leftOutRowIdx = 0; leftOutRowIdx < numInstances; leftOutRowIdx++)
    {
        const int predictedClass = nearest_neighbor(leftOutRowIdx, features);
        const int expectedClass = static_cast<int>(dataset.at(leftOutRowIdx).at(0));

        if (predictedClass != expectedClass)
        {
            numWrong++;
            // Implement early abandoning
            // If number of wrong
            if (numWrong > wrongLimit)
            {
                break;
            }
        }
    }

    return numInstances - numWrong;
}

// Performs leave-one-out validation but moves the instances
// that were predicted wrong to the top
void Project::reorderDataset(const vector<int> &features)
{
    const int numInstances = dataset.size();

    vector<int> mispredictedInstances;

    for (int leftOutRowIdx = 0; leftOutRowIdx < numInstances; leftOutRowIdx++)
    {
        const int predictedClass = nearest_neighbor(leftOutRowIdx, features);
        const int expectedClass = static_cast<int>(dataset.at(leftOutRowIdx).at(0));

        if (predictedClass != expectedClass)
        {
            mispredictedInstances.push_back(leftOutRowIdx);
        }
    }

    // Swapping works because the mispredicted indexes
    // are guranteed to be in order.
    int swapIdx = 0;
    for (int mispredictedIdx : mispredictedInstances)
    {
        vector<double> temp = std::move(dataset.at(swapIdx));
        dataset.at(swapIdx) = std::move(dataset.at(mispredictedIdx));
        dataset.at(mispredictedIdx) = std::move(temp);
    }
}

// print the local best feature set and its accuracy at each depth of the search
void displayLocal(const vector<int> &localBest, double localmax)
{
    cout << "The feature set best at this depth is { ";
    for (int i = 0; i < localBest.size(); i++)
    {
        cout << localBest.at(i);
        if (i < localBest.size() - 1)
        {
            cout << ", ";
        }
    }
    cout << " } with an accuracy of " << localmax << "%" << endl
         << endl;
}

// print the best feature set and its accuracy at the end of the search
void displayBest(const vector<int> &bestFeatures, double max)
{
    cout << endl
         << "Finished search!! The best feature set is { ";
    for (int i = 0; i < bestFeatures.size(); i++)
    {
        cout << bestFeatures.at(i);
        if (i < bestFeatures.size() - 1)
        {
            cout << ", ";
        }
    }
    cout << " }, with an accuracy of " << max << "%" << endl;
}

// forward selection algorithm for feature selection
// starts with an empty set of features and adds one feature at a time to the set of features
void Project::forward_selection()
{
    const int numFeatures = dataset.at(0).size();
    const int numInstances = dataset.size();
    vector<int> bestFeatures;

    int maxCorrect = 0;
    // loop through all features (except the first one, which is the classification)
    for (int depth = 1; depth < numFeatures; ++depth)
    {
        cout << "Exploring accuracy of NN with " << depth << " features!" << endl;

        if (depth > 1)
        {
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
        // Loop through all features and find the best one to add to the feature set.
        for (int featureNum = 1; featureNum < numFeatures; ++featureNum)
        {
            // If the feature is not in the best feature list, then explore it
            if (find(bestFeatures.begin(), bestFeatures.end(), featureNum) == bestFeatures.end())
            {
                vector<int> currFeatures = bestFeatures;
                currFeatures.push_back(featureNum);

                cout << "Using feature(s) { ";
                for (int featureIdx : bestFeatures)
                {
                    cout << featureIdx << ", ";
                }
                cout << featureNum << " } ";
                // Perform leave-one-out validation on the current feature set to find the accuracy
                const int numCorrect = leaveOneOutValidator(currFeatures, wrongLimit);

                const double accuracy = (100.0 * (numCorrect / static_cast<double>(numInstances - 1)));
                // Print the accuracy of the current feature set
                if (numCorrect <= maxCorrect)
                {
                    cout << "was early abandoned.";
                }
                else
                {
                    cout << "accuracy is " << accuracy << "%";
                }
                cout << endl;
                // If the current accuracy is better than the current max, update the current max and the best feature set at this depth
                if (numCorrect > depthMaxCorrect)
                {
                    depthMaxCorrect = numCorrect;

                    depthBestFeatures = bestFeatures;
                    depthBestFeatures.push_back(featureNum);
                }
            }
        }
        // Print the best feature set at this depth and its accuracy
        displayLocal(depthBestFeatures, (100.0 * (depthMaxCorrect / static_cast<double>(numInstances - 1))));
        if (depthMaxCorrect > maxCorrect)
        {
            bestFeatures = depthBestFeatures;
            maxCorrect = depthMaxCorrect;
        }
        else if (depthMaxCorrect < maxCorrect)
        {
            cout << "Warning! Accuracy has decreased! Continuing search in case of local maxima." << endl;
            break;
        }
    }
    // Print the best feature set and its accuracy
    displayBest(bestFeatures, (100.0 * (maxCorrect / static_cast<double>(numInstances - 1))));
}

// backward elimination algorithm for feature selection (opposite of forward selection)
// starts with all features and removes one feature at a time from the set of features
void Project::backward_elimination()
{
    const int numInstances = dataset.size();
    const int numFeatures = dataset.at(0).size();

    vector<int> bestFeatures(numFeatures - 1);
    std::iota(bestFeatures.begin(), bestFeatures.end(), 1);

    int maxCorrect = 0;
    // loop through all features (except the first one, which is the classification)
    for (int depth = 1; depth < numFeatures; ++depth)
    {
        int depthMaxCorrect = 0;
        vector<int> depthBestFeatures;

        if (depth > 1)
        {
            reorderDataset(bestFeatures);
            cout << "Re-ordered dataset for quicker early abandoning!" << endl;
        }

        const int wrongLimit = numInstances - maxCorrect;
        // Loop through all features and find the best one to remove from the feature set.
        for (const int rmFeatureNum : bestFeatures)
        {
            vector<int> currFeatures;
            currFeatures.reserve(bestFeatures.size() - 1);
            cout << "Using feature(s) {";
            for (int featureNum : bestFeatures)
            {
                if (rmFeatureNum == featureNum)
                {
                    continue;
                }

                cout << " " << featureNum;
                currFeatures.push_back(featureNum);
            }
            cout << " } ";
            // Perform leave-one-out validation on the current feature set to find the accuracy
            const int numCorrect = leaveOneOutValidator(currFeatures, wrongLimit);
            const double accuracy = (100.0 * (numCorrect / static_cast<double>(numInstances - 1)));
            // Print the accuracy of the current feature set
            if (numCorrect <= maxCorrect)
            {
                cout << "was early abandoned.";
            }
            else
            {
                cout << "accuracy is " << accuracy << "%";
            }

            cout << endl;
            // If the current accuracy is better than the current max, update the current max and the best feature set at this depth
            if (numCorrect > depthMaxCorrect)
            {
                depthMaxCorrect = numCorrect;
                depthBestFeatures = currFeatures;
            }
        }
        // Print the best feature set at this depth and its accuracy
        const double depthBestAccuracy = (100.0 * (depthMaxCorrect / static_cast<double>(numInstances - 1)));
        displayLocal(depthBestFeatures, depthBestAccuracy);
        if (depthMaxCorrect > maxCorrect)
        {
            bestFeatures = depthBestFeatures;
            maxCorrect = depthMaxCorrect;
        }
        else if (depthMaxCorrect < maxCorrect)
        {
            cout << "Warning! Accuracy has decreased! Continuing search in case of local maxima." << endl;
            break;
        }
    }
    // Print the best feature set and its accuracy at the end of the algorithm
    const double bestAccuracy = (100.0 * (maxCorrect / static_cast<double>(numInstances - 1)));
    displayBest(bestFeatures, bestAccuracy);
}

// Normalize the data via min-max normalization on each feature.
void Project::normalizeData()
{
    cout << endl
         << "Please wait while I normalize the data ... ";
    const int numFeatures = dataset.at(0).size();
    // Vector of the minimum and maximum for each feature
    vector<pair<double, double>> featureMinMax(numFeatures - 1, {INFINITY, -1 * INFINITY});

    // Go through dataset and find actual min & max for each feature
    for (const vector<double> &row : dataset)
    {
        for (int featureIdx = 1; featureIdx < numFeatures; ++featureIdx)
        {
            const double featureVal = row.at(featureIdx);
            auto &[min, max] = featureMinMax.at(featureIdx - 1);

            if (featureVal < min)
                min = featureVal;
            if (featureVal > max)
                max = featureVal;
        }
    }

    // Go through dataset again and apply min max normalization to each feature
    for (vector<double> &row : dataset)
    {
        for (int featureIdx = 1; featureIdx < numFeatures; ++featureIdx)
        {
            double &featureVal = row.at(featureIdx);
            const auto &[min, max] = featureMinMax.at(featureIdx - 1);
            featureVal = (featureVal - min) / (max - min);
        }
    }

    cout << "Done!" << endl
         << endl;
}

// calculate the accuracy of the dataset using the default rate, which is the percentage of the majority class in the dataset
double Project::defaultRate()
{
    int numClass1 = 0, numClass2 = 0;
    // Go through dataset and count the number of each class
    for (const vector<double> &row : dataset)
    {
        const double class_label = row.at(0);
        if (class_label == 1)
            numClass1++;
        else
            numClass2++;
    }
    // Find the biggest class and calculate the accuracy of the default rate
    int biggestClass = max(numClass1, numClass2);
    double accuracy = 100.0 * (static_cast<double>(biggestClass) / dataset.size());

    return accuracy;
}

// search function to run the forward selection or backward elimination algorithm
// and calculate the stating accuracy of the dataset for both algorithms
void Project::search(int choice)
{
    const int numRows = dataset.size();
    const int numFeatures = dataset.at(0).size();
    // Print the number of features and instances in the dataset
    cout << "This dataset has " << (numFeatures - 1) << " features (not including the class attribute), with "
         << numRows << " instances." << endl;
    // Normalize the data before running the algorithms
    normalizeData();
    // run the forward selection algorithm
    if (choice == 1)
    {
        // Print the accuracy of the dataset using the default rate
        cout
            << "Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of "
            << defaultRate() << "%" << endl
            << endl;
        cout << "Beginning search." << endl
             << endl;
        forward_selection();
    }
    // run the backward elimination algorithm
    else if (choice == 2)
    {
        std::vector<int> allFeatures(numFeatures - 1);
        std::iota(allFeatures.begin(), allFeatures.end(), 1);
        // print the accuracy of the dataset using all features before running the algorithm
        cout
            << "Running nearest neighbor with ALL features, using \"leaving-one-out\" evaluation, I get an accuracy of "
            << 100 * (static_cast<double>(leaveOneOutValidator(allFeatures, numRows)) / (numRows - 1)) << "%" << endl
            << endl;
        cout << "Beginning search." << endl
             << endl;
        backward_elimination();
    }
    else
    {
        cout << "Error! Not a correct selection from options" << endl;
    }
}