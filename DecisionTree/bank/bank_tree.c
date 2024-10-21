#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define NUM_ATTRIBUTES 16
#define MAX_VAL 12 
float thresholds[NUM_ATTRIBUTES] = {38, 0, 0, 0, 0, 452.5, 0, 0, 0, 16, 0, 180, 2, -1, 0, 0};
bool isNumeric[NUM_ATTRIBUTES] = {true, false, false, false, false, true, false, false, false, true, false, true, true, true, true, false};

int countEntries(char fileName[]);
int importData(char fileName[], int data[][NUM_ATTRIBUTES+1], int numInstances, int numAttributes);
int importTree(char fileName[], int tree[][17], int numInstances, int numAttributes);
int valueToInt(char* value, int attribute);

typedef struct
{
    int id;
    int attribute;
    int value;
    int label;
    int parent;
    int leaf[MAX_VAL];
}Branch;

int main()
{
    //Get input file names
    char userInput[50];
    printf("Enter data file name: \n\n");
    scanf(" %s", userInput);
    printf("\n");
    //Import data from CSV
    int numInstances = countEntries(userInput);
    if (numInstances == -1)
    {
        return 1;
    }
    int data[numInstances][NUM_ATTRIBUTES+1];
    importData(userInput, data, numInstances, NUM_ATTRIBUTES+1);
    int dataLabels[numInstances];
    int myLabels[numInstances];
    //Set up label arrays for error calculations
    for (int i = 0; i < numInstances; i++)
    {
        dataLabels[i] = data[i][NUM_ATTRIBUTES];
        myLabels[i] = -3;
    }

    //Import tree from CSV
    printf("Enter tree file name: \n\n");
    scanf(" %s", userInput);
    printf("\n");
    int numBranches = countEntries(userInput);
    if (numBranches == -1)
    {
        return 1;
    }
    int tempTree[numBranches][17];
    Branch tree[numBranches];
    importTree(userInput, tempTree, numBranches, 17);
    for (int i = 0; i < numBranches; i++)
    {
        tree[i].id = tempTree[i][0];
        tree[i].attribute = tempTree[i][1];
        tree[i].value = tempTree[i][2];
        tree[i].label = tempTree[i][3];
        tree[i].parent = tempTree[i][4];
        for (int j = 0; j < MAX_VAL; j++)
        {
            tree[i].leaf[j] = tempTree[i][5+j];
        }
    }

    //Put data through decision tree
    int branchIndex = 0;
    int instanceIndex = 0;
    //For each instance in dataset
    while (instanceIndex < numInstances)
    {   
        //If the current branch in the tree has a label, assign that label to that instance
        if (tree[branchIndex].label > -1)
        {
            myLabels[instanceIndex] = tree[branchIndex].label;
            branchIndex = 0;
            instanceIndex++;
        }
        //If the current branch does not have a label, move to the leaf whose value on the split attribute matches the instance
        else
        {
            for (int j = 0; j < MAX_VAL; j++)
            {
                if (data[instanceIndex][tree[branchIndex].attribute] == tree[tree[branchIndex].leaf[j]].value)
                {
                    branchIndex = tree[branchIndex].leaf[j];
                    break;
                }
            }
            
        }
    }
    char ans;
    printf("Print labels in terminal? (Y/N)\n\n");
    scanf(" %c", &ans);
    printf("\n");
    if (ans == 'Y')
    {
        printf("Real Label -- Predicted Label\n");
        for (int i = 0; i < numInstances; i++)
        {
            printf("\t%d  --  %d\n", dataLabels[i], myLabels[i]);
        }
    }

    int incorrectPredictions = 0;
    for (int i = 0; i < numInstances; i++)
    {
        if (dataLabels[i] != myLabels[i])
        {
            incorrectPredictions++;
        }
    }

    printf("\n%d incorrect predictions on %d instances\n", incorrectPredictions, numInstances);
    printf("prediction error: %.2f%% \n\n", ((float) incorrectPredictions / (float) numInstances)*100);

}

int countEntries(char fileName[])
{
     int count = 0;

    //Open input file
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file: %s\n", fileName);
        return -1;
    }

    char row[100];

    //Count number of instances in input file
    while (feof(inputFile) != true)
    {
        if (fgets(row, 100, inputFile) == NULL)
        {
            fclose(inputFile);
            return count;
        }
        else
        {
            count++;
        }
    }
    fclose(inputFile);
    return count;
}

int importData(char fileName[], int data[][NUM_ATTRIBUTES+1], int numInstances, int numAttributes)
{
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file: %s\n", fileName);
        return -1;
    }

    char row[100];
    char *token;

    //Parse input CSV into data instance struct array
    while (feof(inputFile) != true)
    {
        for (int i = 0; i < numInstances; i++)
        {
            if (fgets(row, 100, inputFile) == NULL)
            {
                fclose(inputFile);
                return 0;
            }
            else
            {
                token = strtok(row, ",");
                for (int j = 0; j < numAttributes; j++)
                {
                    data[i][j] = valueToInt(token, j);
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

int importTree(char fileName[], int tree[][17], int numInstances, int numAttributes)
{
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file: %s\n", fileName);
        return -1;
    }

    char row[100];
    char *token;

    //Parse input CSV into data instance struct array
    while (feof(inputFile) != true)
    {
        for (int i = 0; i < numInstances; i++)
        {
            if (fgets(row, 100, inputFile) == NULL)
            {
                fclose(inputFile);
                return 0;
            }
            else
            {
                token = strtok(row, ",");
                for (int j = 0; j < numAttributes; j++)
                {
                    tree[i][j] = valueToInt(token, -2);
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

//Convert value strings from input dataset to integers
int valueToInt(char* value, int attribute)
{
    if (attribute == -2)
    {
        return atoi(value);
    }
    else if (isNumeric[attribute])
    {
        if (atoi(value) >= thresholds[attribute])
        {
            //yes >=
            return 0;
        }
        else
        {
            //no <
            return 1;
        }
    }
    else
    {
        switch (attribute)
        {
            case 1:
                if (!strcmp(value, "admin"))
                    return 0;
                else if (!strcmp(value, "unknown"))
                    return 1;
                else if (!strcmp(value, "unemployed"))
                    return 2;
                else if (!strcmp(value, "management"))
                    return 3;
                else if (!strcmp(value, "housemaid"))
                    return 4;
                else if (!strcmp(value, "entrepreneur"))
                    return 5;
                else if (!strcmp(value, "student"))
                    return 6;
                else if (!strcmp(value, "blue-collar"))
                    return 7;
                else if (!strcmp(value, "self-employed"))
                    return 8;
                else if (!strcmp(value, "retired"))
                    return 9;
                else if (!strcmp(value, "technician"))
                    return 10;
                else if (!strcmp(value, "services"))
                    return 11;
                break;
            case 2:
                if (!strcmp(value, "married"))
                    return 0;
                else if (!strcmp(value, "divorced"))
                    return 1;
                else if (!strcmp(value, "single"))
                    return 2;
                break;
            case 3:
                if (!strcmp(value, "unknown"))
                    return 0;
                else if (!strcmp(value, "secondary"))
                    return 1;
                else if (!strcmp(value, "primary"))
                    return 2;
                else if (!strcmp(value, "tertiary"))
                    return 3;
                break;
            case 4:
                if (!strcmp(value, "yes"))
                    return 0;
                else if (!strcmp(value, "no"))
                    return 1;
                break;
            case 6:
                if (!strcmp(value, "yes"))
                    return 0;
                else if (!strcmp(value, "no"))
                    return 1;
                break;
            case 7:
                if (!strcmp(value, "yes"))
                    return 0;
                else if (!strcmp(value, "no"))
                    return 1;
                break;
            case 9:
                if (!strcmp(value, "unknown"))
                    return 0;
                else if (!strcmp(value, "telephone"))
                    return 1;
                else if (!strcmp(value, "cellular"))
                    return 2;
                break;
            case 10:
                if (!strcmp(value, "jan"))
                    return 0;
                else if (!strcmp(value, "feb"))
                    return 1;
                else if (!strcmp(value, "mar"))
                    return 2;
                if (!strcmp(value, "apr"))
                    return 3;
                else if (!strcmp(value, "may"))
                    return 4;
                else if (!strcmp(value, "jun"))
                    return 5;
                if (!strcmp(value, "jul"))
                    return 6;
                else if (!strcmp(value, "aug"))
                    return 7;
                else if (!strcmp(value, "sep"))
                    return 8;
                if (!strcmp(value, "oct"))
                    return 9;
                else if (!strcmp(value, "nov"))
                    return 10;
                else if (!strcmp(value, "dec"))
                    return 11;
                break;
            case 15:
                if (!strcmp(value, "unknown"))
                    return 0;
                else if (!strcmp(value, "other"))
                    return 1;
                else if (!strcmp(value, "failure"))
                    return 2;
                else if (!strcmp(value, "success"))
                    return 3;
                break;
            case 16:
                if (!strcmp(value, "yes"))
                    return 0;
                else if (!strcmp(value, "no"))
                    return 1;
                break;
        }
    }
    return -1;
}
