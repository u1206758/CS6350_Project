#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define NUM_ATTRIBUTES 4
#define MAX_VAL 3 

#define DATA_FILE "tennis.csv"
#define TREE_FILE "tennis_tree.csv"

int countEntries(char fileName[]);
int importData(char fileName[], int data[][NUM_ATTRIBUTES+1], int numInstances, int numAttributes);
int importTree(char fileName[], int tree[][8], int numInstances, int numAttributes);
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
    //Import data from CSV
    int numInstances = countEntries(DATA_FILE);
    if (numInstances == -1)
    {
        return 1;
    }
    int data[numInstances][NUM_ATTRIBUTES+1];
    importData(DATA_FILE, data, numInstances, NUM_ATTRIBUTES+1);
    int dataLabels[numInstances];
    int myLabels[numInstances];
    //Set up label arrays for error calculations
    for (int i = 0; i < numInstances; i++)
    {
        dataLabels[i] = data[i][NUM_ATTRIBUTES];
        myLabels[i] = -3;
    }

    //Import tree from CSV
    int numBranches = countEntries(TREE_FILE);
    int tempTree[numBranches][8];
    Branch tree[numBranches];
    importTree(TREE_FILE, tempTree, numBranches, 8);
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
    printf("Real Label -- Predicted Label\n");
    for (int i = 0; i < numInstances; i++)
    {
        printf("\t%d  --  %d\n", dataLabels[i], myLabels[i]);
    }

    int incorrectPredictions = 0;
    for (int i = 0; i < numInstances; i++)
    {
        if (dataLabels[i] != myLabels[i])
        {
            incorrectPredictions++;
        }
    }

    printf("%d incorrect predictions on %d instances\n", incorrectPredictions, numInstances);
    printf("prediction error: %.2f%% \n", ((float) incorrectPredictions / (float) numInstances)*100);

}

int countEntries(char fileName[])
{
     int count = 0;

    //Open input file
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file");
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
        printf("Error opening file");
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
                    data[i][j] = valueToInt(token, 0);
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

int importTree(char fileName[], int tree[][8], int numInstances, int numAttributes)
{
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file");
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
                    tree[i][j] = valueToInt(token, -1);
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
    //branchIndex, attribute, value, label, parent, MAX_VAL # of leaves
    int dec;
    if (attribute == -1)
    {
        return atoi(value);
    }

    /*
        data[inst][0] - outlook
            sunny - 0
            overcast - 1
            rainy - 2
        data[]inst[1] - temperature
            hot - 0
            medium - 1
            cool - 2
        data[inst][2] - humidity
            high - 0
            normal - 1
            low - 2
        data[inst][3] - wind
            strong - 0
            weak - 1
        data[inst][4] - play
            no - 0
            yes - 1
    */   

    if (!strcmp(value, "sunny") || !strcmp(value, "hot") || !strcmp(value, "high") || !strcmp(value, "strong") || !strcmp(value, "no"))
        return 0;
    else if (!strcmp(value, "overcast") || !strcmp(value, "medium") || !strcmp(value, "normal") || !strcmp(value, "weak") || !strcmp(value, "yes"))
        return 1;
    else if (!strcmp(value, "rainy") || !strcmp(value, "cool") || !strcmp(value, "low"))
        return 2;
    else
        return -1;
}
