#ifndef PROJECT2_PROJECT_H
#define PROJECT2_PROJECT_H

#include <vector>

using namespace std;
class Project {         
    public:
        Project(vector<vector<double>> dataset);
        void search(int choice);
    private:
        double defaultRate();
        void normalizeData();
        void reorderDataset(const vector<int>& features);
        int leaveOneOutValidator(const vector<int>& features, int wrongLimit);
        int nearest_neighbor(const int queryIdx, const vector<int>& features);
        void forward_selection();
        void backward_elimination();

        vector<vector<double>> dataset;
};

#endif