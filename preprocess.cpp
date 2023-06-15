#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int main()
{
    ifstream inputFile("../data/breast-cancer.csv");
    ofstream outputFile("../data/processed_breast-cancer.csv");
    string line;

    // Skip the first row
    getline(inputFile, line);

    while (getline(inputFile, line))
    {
        istringstream iss(line);
        string value;

        // Skip first column
        getline(iss, value, ',');

        vector<std::string> newRow;

        while (getline(iss, value, ','))
        {
            if (newRow.empty())
            { // This is the second column
                if (value == "M" || value == "m")
                {
                    newRow.push_back("1");
                }
                else if (value == "b" || value == "B")
                {
                    newRow.push_back("0");
                }
                else
                {
                    newRow.push_back(value);
                }
            }
            else
            {
                newRow.push_back(value);
            }
        }

        // Write the new row to the output file
        for (size_t i = 0; i < newRow.size(); ++i)
        {
            outputFile << newRow[i];
            if (i != newRow.size() - 1)
            {
                outputFile << " ";
            }
        }
        outputFile << "\n";
    }

    inputFile.close();
    outputFile.close();
}
