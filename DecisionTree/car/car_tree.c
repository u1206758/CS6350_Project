#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define NUM_ATTRIBUTES 6
#define MAX_VAL 4 

int countEntries(char fileName[]);
int importData(char fileName[], int data[][NUM_ATTRIBUTES+1], int numInstances, int numAttributes);
int importTree(char fileName[], int tree[][9], int numInstances, int numAttributes);
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
    int tempTree[numBranches][9];
    Branch tree[numBranches];
    importTree(userInput, tempTree, numBranches, 9);
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

int importTree(char fileName[], int tree[][9], int numInstances, int numAttributes)
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
        if ((char )value[0] == '-')
        {
            dec = -1 * ((char) value[1] - 48);
        }
        else
        {
            dec = (char) value[0] - 48;
        }
        return dec;
    }

       /*
        data[inst][0] - buying
            vhigh - 0
            high - 1
            med - 2
            low - 3
        data[]inst[1] - maint
            vhigh - 0
            high - 1
            med - 2
            low - 3
        data[inst][2] - doors
            2 - 0
            3 - 1
            4 - 2
            5more - 3
        data[inst][3] - persons
            2 - 0
            4 - 1
            more - 2
        data[inst][4] - lug_boot
            small - 0
            med - 1
            big - 2
        data[inst][5] - safety
            low - 0
            med - 1
            high - 2
        data[inst][6] - label
            unacc - 0
            acc - 1
            good - 2
            vgood - 3
    */

   switch (attribute)
   {
        case 0:
            if (!strcmp(value, "vhigh"))
                return 0;
            else if (!strcmp(value, "high"))
                return 1;
            else if (!strcmp(value, "med"))
                return 2;
            else if (!strcmp(value, "low"))
                return 3;
            break;
        case 1:
            if (!strcmp(value, "vhigh"))
                return 0;
            else if (!strcmp(value, "high"))
                return 1;
            else if (!strcmp(value, "med"))
                return 2;
            else if (!strcmp(value, "low"))
                return 3;
            break;
        case 2:
            if (!strcmp(value, "2"))
                return 0;
            else if (!strcmp(value, "3"))
                return 1;
            else if (!strcmp(value, "4"))
                return 2;
            else if (!strcmp(value, "5more"))
                return 3;
            break;
        case 3:
            if (!strcmp(value, "2"))
                return 0;
            else if (!strcmp(value, "4"))
                return 1;
            else if (!strcmp(value, "more"))
                return 2;
            break;
        case 4:
            if (!strcmp(value, "small"))
                return 0;
            else if (!strcmp(value, "med"))
                return 1;
            else if (!strcmp(value, "big"))
                return 2;
            break;
        case 5:
            if (!strcmp(value, "low"))
                return 0;
            else if (!strcmp(value, "med"))
                return 1;
            else if (!strcmp(value, "high"))
                return 2;
            break;
        case 6:
            if (!strcmp(value, "unacc"))
                return 0;
            else if (!strcmp(value, "acc"))
                return 1;
            else if (!strcmp(value, "good"))
                return 2;
            else if (!strcmp(value, "vgood"))
                return 3;
            break;
   }
   return -1;
}
