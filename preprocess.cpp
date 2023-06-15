#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int main()
{
    // Open the input and output files
    ifstream inputFile("../data/breast-cancer.csv");
    ofstream outputFile("../data/processed_breast-cancer.csv");
    string line;

    // Skip the first row (column names) of the input file
    getline(inputFile, line);

    // Read the input file line by line
    while (getline(inputFile, line))
    {
        istringstream iss(line);
        string value;

        // Skip first column (id)
        getline(iss, value, ',');

        vector<std::string> newRow;

        while (getline(iss, value, ','))
        {
            if (newRow.empty())
            { // This is the second column (class) of the input file, map to 0 or 1
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
    // Close the input and output files
    inputFile.close();
    outputFile.close();
}
