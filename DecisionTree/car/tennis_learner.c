//This program implements a decision tree learning algorithm for the car evaluation task in HW1 part 2 problem 2.
//Users can select information gain, majority error, or gini index, and optionally set a maximum depth from 1 to 6.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define NUM_I 14
#define NUM_LABELS 2
#define NUM_ATTRIBUTES 4
int numValues[NUM_ATTRIBUTES] = {3, 3, 3, 2};
#define MAX_VAL 3 
#define MAX_BRANCH 30

int splitLeaf(short currentInstances[NUM_I], int data[][NUM_ATTRIBUTES+1], int numInstances, int method, bool parentAttribute[NUM_ATTRIBUTES], int branchIndex);
float ig_initial(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances);
float ig_gain(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances, int attribute);
float me_initial(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances);
float me_gain(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances, int attribute);
float gini_initial(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances);
float gini_gain(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances, int attribute);

int countData(void);
int importData(int data[][NUM_ATTRIBUTES+1], int numInstances);
int valueToInt(char* value);
int getMethod(void);
int getMaxDepth(void);

typedef struct
{
    bool active;    //Whether this branch slot is used by the tree
    int level;      //The depth in the tree this branch is on
    int parent;     //The ID of the parent branch
    int leaf[MAX_VAL];    //The IDs of the leaves of this branch
    int attribute;  //The attribute this branch is split on (if any)
    int value;      //The value a leaf represents when split from parent branch
    int label;      //The label for this branch (-1 no label, -2 all children labelled)
}Branch;

void printTree(Branch tree[], int maxBranches);
void decodeAttribute(int attribute);
void decodeValue(int attribute, int value);
void decodeLabel(int label);
void exportTree(Branch tree[], int maxBranches);

int getNextID(Branch tree[], int maxBranches);

int main()
{
    //Import data from CSV
    int numInstances = countData();
    if (numInstances == -1)
    {
        return 1;
    }
    int data[numInstances][NUM_ATTRIBUTES+1];
    importData(data, numInstances);
    int method = getMethod();
    int maxDepth = getMaxDepth();
    int maxBranches = MAX_BRANCH;
    Branch tree[maxBranches];
    int currentLevel = 1;
    bool allDone = false;

    //Initialize all possible branches
    for (int i = 0; i < maxBranches; i++)
    {
        tree[i].active = false;
        tree[i].level = -1;
        tree[i].parent = -1;
        for (int j = 0; j < MAX_VAL; j++)
            tree[i].leaf[j] = -1;
        tree[i].attribute = -1;
        tree[i].value = -1;
        tree[i].label = -1;
    }
    //Initialize head of tree
    short currentInstances[maxBranches][numInstances];

    for (int i = 0; i < maxBranches; i++)
    {
        for (int j = 0; j < numInstances; j++)
        {
            if (i == 0)
            {
                currentInstances[i][j] = j;
            }
            else
            {
                currentInstances[i][j] = -1;
            }
        }
    }
    int branchIndex = 0;
    tree[0].active = true;
    tree[0].level = 1;
    while(!allDone)
    {
        int lastLabel = -1;
        bool readyToLabel = true;
        bool allLeavesLabelled = true;
        bool hasLeaves = false;
        int labelCount[NUM_LABELS];
        for (int i = 0; i < NUM_LABELS; i++)
        {
            labelCount[i] = 0;
        }
        int maxLabelCount = 0;
        int maxLabel = -1;
        bool parentAttribute[NUM_ATTRIBUTES];
        int parentValue[NUM_ATTRIBUTES];
        for (int i = 0; i < NUM_ATTRIBUTES; i++)
        {
            parentAttribute[i] = false;
            parentValue[i] = -1;
        }

        //If current branch/leaf is at max level
        if (tree[branchIndex].level > maxDepth)
        {
            //find most common label for current value
            for (int i = 0; i < numInstances; i++)
            {
                if (data[i][tree[tree[branchIndex].parent].attribute] == tree[branchIndex].value)
                {
                    for (int j = 0; j < NUM_LABELS; j++)
                    {
                        if (data[i][NUM_ATTRIBUTES] == j)
                        {
                            labelCount[j]++;
                        }
                    }
                }
            }
            for (int i = 0; i < NUM_LABELS; i++)
            {
                if (labelCount[i] > maxLabelCount)
                {
                    maxLabelCount = labelCount[i];
                    maxLabel = i;
                }
            }
            //If branch value has no instances
            if (maxLabel == -1)
            {
                for (int i = 0; i < NUM_LABELS; i++)
                {
                    labelCount[i] = 0;
                }
                maxLabelCount = 0;
                //find most common label for current value
                for (int i = 0; i < numInstances; i++)
                {
                    if (data[i][tree[tree[tree[branchIndex].parent].parent].attribute] == tree[tree[branchIndex].parent].value)
                    {
                        for (int j = 0; j < NUM_LABELS; j++)
                        {
                            if (data[i][NUM_ATTRIBUTES] == j)
                            {
                                labelCount[j]++;
                            }
                        }
                    }
                }
                for (int i = 0; i < NUM_LABELS; i++)
                {
                    if (labelCount[i] > maxLabelCount)
                    {
                        maxLabelCount = labelCount[i];
                        maxLabel = i;
                    }
                }
            }
            //Assign most common label to leaf
            tree[branchIndex].label = maxLabel;

            //Check if all leaves of parent branch are labelled
            for (int j = 0; j < numValues[tree[tree[branchIndex].parent].attribute]; j++)
            {
                //If not, set flag false
                if (tree[tree[tree[branchIndex].parent].leaf[j]].label == -1)
                {
                    allLeavesLabelled = false;
                    break;
                }
            }                 
            //Set index back to head if all labelled
            if (allLeavesLabelled)
            {
                tree[tree[branchIndex].parent].label = -2;  //mark current branch as having all leaves labelled
            }

            //set index to parent
            branchIndex = tree[branchIndex].parent;
        }
        else
        {
            //Check if all instances have same label
            for (int i = 0; i < numInstances; i++)
            {   
                //If current instance is in current set
                if (currentInstances[branchIndex][i] != -1)
                {
                    //Check if all labels are the same
                    if (lastLabel == -1)
                    {
                        lastLabel = data[i][NUM_ATTRIBUTES];
                    }
                    else
                    {
                        if (lastLabel != data[i][NUM_ATTRIBUTES])
                        {
                            readyToLabel = false;
                            break;
                        }
                        else
                        {
                            lastLabel = data[i][NUM_ATTRIBUTES];
                        }
                    }
                }
            }
            //If branch is ready to label
            if (readyToLabel)
            {
                //Label branch
                //If branch value has no instances
                if (lastLabel == -1)
                {
                    //find most common label for current value
                    for (int i = 0; i < numInstances; i++)
                    {
                        if (data[i][tree[tree[tree[branchIndex].parent].parent].attribute] == tree[tree[branchIndex].parent].value)
                        {
                            for (int j = 0; j < NUM_LABELS; j++)
                            {
                                if (data[i][NUM_ATTRIBUTES] == j)
                                {
                                    labelCount[j]++;
                                }
                            }
                        }
                    }
                    for (int i = 0; i < NUM_LABELS; i++)
                    {
                        if (labelCount[i] > maxLabelCount)
                        {
                            maxLabelCount = labelCount[i];
                            maxLabel = i;
                        }
                    }
                    //set lastLabel to maxLabel
                    lastLabel = maxLabel;
                }
                tree[branchIndex].label = lastLabel;
                //If index at head, all done, else set index to parent
                if (branchIndex == 0)
                {
                    allDone = true;
                }
                else
                {
                    branchIndex = tree[branchIndex].parent;    
                }
            
                //Check if all leaves of current branch are labelled
                for (int j = 0; j < numValues[tree[branchIndex].attribute]; j++)
                {
                    //If not, set index to next unlabelled leaf
                    if (tree[tree[branchIndex].leaf[j]].label == -1)
                    {
                        allLeavesLabelled = false;
                        branchIndex = tree[branchIndex].leaf[j];
                        break;
                    }
                }                 
                //Set index back to head if all labelled
                if (allLeavesLabelled)
                {
                    tree[branchIndex].label = -2;  //mark current branch as having all leaves labelled
                    branchIndex = tree[branchIndex].parent;
                }
            }   
            else //branch is not ready to label
            {
                //Check if leaves exist on this branch
                for (int i = 0; i < numValues[tree[branchIndex].attribute]; i++)
                {
                    if (tree[branchIndex].leaf[i != -1])
                        {
                            hasLeaves = true;
                            break;
                        }
                }
                if (hasLeaves)
                {
                    //Check if all leaves are labelled
                    for (int i = 0; i < numValues[tree[branchIndex].attribute]; i++)
                    {
                        //if all are not labelled, set index to next unlabeleld
                        if (tree[tree[branchIndex].leaf[i]].label == -1)
                        {
                            allLeavesLabelled = false;
                            branchIndex = tree[branchIndex].leaf[i];
                            break;
                        }
                    }
                    if (allLeavesLabelled)
                    {
                        //if all leaves on all branches under head are labelled, done
                        if (branchIndex == 0)
                        {
                            allDone = true;
                        }
                        else // set index back to parent
                        {
                            tree[branchIndex].label = -2;
                            branchIndex = tree[branchIndex].parent;
                        }
                    }
                }
                else //no leaves, not ready to label, need to split
                {
                    //Find all attributes already split in the current path
                    int tempIndex = branchIndex;
                    while (tempIndex > -1)
                    {
                        if (tempIndex > 0)
                        {
                            parentValue[tree[tree[tempIndex].parent].attribute] = tree[tempIndex].value;
                        }
                        tempIndex = tree[tempIndex].parent;
                        parentAttribute[tree[tempIndex].attribute] = true;
                    }
                    //split
                    tree[branchIndex].attribute = splitLeaf(currentInstances[branchIndex], data, numInstances, method, parentAttribute, branchIndex);
                    //create leaves & assign values
                    for (int i = 0; i < numValues[tree[branchIndex].attribute]; i++)
                    {
                        tree[branchIndex].leaf[i] = getNextID(tree, maxBranches);
                        if (tree[branchIndex].leaf[i] == -1)
                        {
                            printf("ERROR: out of branches!\n");
                            printf("%d\n", branchIndex);
                            return 1;
                        }
                        tree[tree[branchIndex].leaf[i]].active = true;
                        tree[tree[branchIndex].leaf[i]].value = i;
                        tree[tree[branchIndex].leaf[i]].parent = branchIndex;
                        tree[tree[branchIndex].leaf[i]].level = tree[branchIndex].level + 1;
                        //for each instance
                        for (int j = 0; j < numInstances; j++)
                        {
                            //if the instance value mathing the leaf value for the parent attribute is also present in parent subset
                            if (data[j][tree[branchIndex].attribute] == i && currentInstances[branchIndex][j] != -1)
                            {
                                currentInstances[tree[branchIndex].leaf[i]][j] = j;
                            }
                            else
                            {
                                currentInstances[tree[branchIndex].leaf[i]][j] = -1;
                            }
                        }
                    }
                    //set branch index to first leaf
                    branchIndex = tree[branchIndex].leaf[0];
                }
            }
        }
    }

    printTree(tree, maxBranches);
    exportTree(tree, maxBranches);
    return 0;
}

int splitLeaf(short currentInstances[NUM_I], int data[][NUM_ATTRIBUTES+1], int numInstances, int method, bool parentAttribute[NUM_ATTRIBUTES], int branchIndex)
{
    //Main algorithm loop
    float initialInformation;
    float attributeGain[NUM_ATTRIBUTES];
    float bestGain = -1;
    int bestAttribute = -1;
    
    //Calculate initial label information
    switch (method)
    {
        //IG
        case 0:
            initialInformation = ig_initial(currentInstances, data, numInstances);
            break;
        //ME
        case 1:
            initialInformation = me_initial(currentInstances, data, numInstances);
            break;
        //Gini
        case 2:
            initialInformation = gini_initial(currentInstances, data, numInstances);
            break;
    }
    for (int i = 0; i < NUM_ATTRIBUTES; i++)
    {
        if (!parentAttribute[i])
        {
            switch (method)
            {
                //IG
            case 0:
                attributeGain[i] = ig_gain(currentInstances, data, numInstances, i);
                break;
            //ME
            case 1:
                attributeGain[i] = me_gain(currentInstances, data, numInstances, i);
                break;
            //Gini
            case 2:
                attributeGain[i] = gini_gain(currentInstances, data, numInstances, i);
                break;
            }
        }
        else
        {
            attributeGain[i] = 10;
        }
    }
    
    //Find max gain attribute
    for (int i = 0; i < NUM_ATTRIBUTES; i++)
    {
        if (initialInformation - attributeGain[i] > bestGain)
        {
            bestGain = initialInformation - attributeGain[i];
            bestAttribute = i;
        }
    }
    return bestAttribute;
}

//Calculate entropy on labels for current set of instances
float ig_initial(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances)
{
    float totalCount = 0;
    float labelCount[NUM_LABELS];
    float init = 0;
    for (int i = 0; i < NUM_LABELS; i ++)
    {
        labelCount[i] = 0;
    }

    for (int i = 0; i < numInstances; i++)
    {
        if (subset[i] != -1)
        {
            totalCount++;
            for (int j = 0; j < NUM_LABELS; j++)
            {
                if (dataset[i][NUM_ATTRIBUTES] == j)
                {
                    labelCount[j]++;
                }       
            }
        }
    }

    for (int i = 0; i < NUM_LABELS; i++)
    {
        if (labelCount[i] != 0)
        {
            init -= ((labelCount[i]/totalCount)*log2(labelCount[i]/totalCount));
        }
    }

    return init;
}

//Calculate weighted entropy gain for each attribute in current instance set
float ig_gain(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances, int attribute)
{
    float totalCount = 0;
    float valueCount[numValues[attribute]];
    float labelCount[numValues[attribute]][NUM_LABELS];
    float entropy[numValues[attribute]];
    float weightedEntropy = 0;

    //Initialize values to zero
    for (int j = 0; j < numValues[attribute]; j++)
    {
        valueCount[j] = 0;
        for (int k = 0; k < NUM_LABELS; k++)
        {
            labelCount[j][k] = 0;
        }
        entropy[j] = 0;
    }

    //for each instance that is in the current subset
    for (int i = 0; i < numInstances; i++)
    {
        if (subset[i] != -1)
        {
            //count values for each attribute and their label
            for (int j = 0; j < numValues[attribute]; j++)
            {
                if (dataset[i][attribute] == j)
                {
                    valueCount[j]++;
                    totalCount++;
                    for (int k = 0; k < NUM_LABELS; k++)
                    {
                        if (dataset[i][NUM_ATTRIBUTES] == k)
                        {
                            labelCount[j][k]++;
                        }
                    }
                }
            }
        }
    }
    


    for (int j = 0; j < numValues[attribute]; j++)
    {
        for (int k = 0; k < NUM_LABELS; k++)
        {
            if (labelCount[j][k] == 0)
            {
                entropy[j] = 0;
            }
            else
            {
                entropy[j] -= ((labelCount[j][k]/valueCount[j])*log2(labelCount[j][k]/valueCount[j]));
            }
        }
        weightedEntropy += valueCount[j]/totalCount*entropy[j];
    }
    return weightedEntropy;
}

//Calculate majority error on labels for current set of instances
float me_initial(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances)
{
    float totalCount = 0;
    float labelCount[NUM_LABELS];
    for (int i = 0; i < NUM_LABELS; i++)
    {
        labelCount[i] = 0;
    }
    for (int i = 0; i < numInstances; i++)
    {
        if (subset[i] != -1)
        {
            totalCount++;
            for (int j = 0; j < NUM_LABELS; j++)
            {
                if (dataset[i][NUM_ATTRIBUTES] == j)
            {
                labelCount[j]++;
            }
            }
        }
    }

    float max = -1;

    for (int j = 0; j < NUM_LABELS; j++)
    {
        if (labelCount[j] > max)
        {
            max = labelCount[j];
        }
    }
    return 1 - (max/totalCount);
}

//Calculate weighted majority error gain for each attribute in current instance set
float me_gain(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances, int attribute)
{
    float totalCount = 0;
    float valueCount[numValues[attribute]];
    float labelCount[numValues[attribute]][NUM_LABELS];
    float me[numValues[attribute]];
    float weightedme = 0;

    //Initialize values to zero
    for (int j = 0; j < numValues[attribute]; j++)
    {
        valueCount[j] = 0;
        for (int k = 0; k < NUM_LABELS; k++)
        {
            labelCount[j][k] = 0;
        }
        me[j] = 0;
    }

    //for each instance that is in the current subset
    for (int i = 0; i < numInstances; i++)
    {
        if (subset[i] != -1)
        {
            //count values for each attribute and their label
            for (int j = 0; j < numValues[attribute]; j++)
            {
                if (dataset[i][attribute] == j)
                {
                    valueCount[j]++;
                    totalCount++;
                    for (int k = 0; k < NUM_LABELS; k++)
                    {
                        if (dataset[i][NUM_ATTRIBUTES] == k)
                        {
                            labelCount[j][k]++;
                        }
                    }
                }
            }
        }
    }
    
    float max[numValues[attribute]];
    for (int i = 0; i < numValues[attribute]; i++)
    {
        max[i] = -1;
    }

    for (int j = 0; j < numValues[attribute]; j++)
    {
        if (valueCount[j] == 0)
        {
            me[j] = 1;
        }
        else
        {
            for (int k = 0; k < NUM_LABELS; k++)
            {
                if (labelCount[j][k] > max[j])
                {
                    max[j] = labelCount[j][k];
                }
                me[j] = 1 - (max[j]/valueCount[j]);
            }
        }
        weightedme += valueCount[j]/totalCount*me[j];
    }
    return weightedme;
}

//Calculate gini on labels for current set of instances
float gini_initial(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances)
{
    float totalCount = 0;
    float labelCount[NUM_LABELS];
    float gini = 0;
    for (int i = 0; i < NUM_LABELS; i++)
    {
        labelCount[i] = 0;
    }
    for (int i = 0; i < numInstances; i++)
    {
        if (subset[i] != -1)
        {
            totalCount++;
            for (int j = 0; j < NUM_LABELS; j++)
            {
                if (dataset[i][NUM_ATTRIBUTES] == j)
                {
                    labelCount[j]++;
                }
            }
        }
    }

    for (int i = 0; i < NUM_LABELS; i++)
    {
        gini += (labelCount[i]/totalCount) * (labelCount[i]/totalCount);
    }

    return 1 - gini;
}

//Calculate weighted gini gain for each attribute in current instance set
float gini_gain(short subset[], int dataset[][NUM_ATTRIBUTES+1], int numInstances, int attribute)
{
    float totalCount = 0;
    float valueCount[numValues[attribute]];
    float labelCount[numValues[attribute]][NUM_LABELS];
    float giniIntermediate[numValues[attribute]];
    float gini[numValues[attribute]];
    float weightedGini = 0;

    //Initialize values to zero
    for (int j = 0; j < numValues[attribute]; j++)
    {
        valueCount[j] = 0;
        for (int k = 0; k < NUM_LABELS; k++)
        {
            labelCount[j][k] = 0;
        }
        giniIntermediate[j] = 0;
        gini[j] = 0;
    }

    //for each instance that is in the current subset
    for (int i = 0; i < numInstances; i++)
    {
        if (subset[i] != -1)
        {
            //count values for each attribute and their label
            for (int j = 0; j < numValues[attribute]; j++)
            {

                if (dataset[i][attribute] == j)
                {
                    valueCount[j]++;
                    totalCount++;
                    for (int k = 0; k < NUM_LABELS; k++)
                    {
                        if (dataset[i][NUM_ATTRIBUTES] == k)
                        {
                            labelCount[j][k]++;
                        }                     
                    }
                }
            }
        }
    }
    


    for (int j = 0; j < numValues[attribute]; j++)
    {
        if (valueCount[j] == 0)
        {
            gini[j] = 1;
        }
        else
        {
            for (int k = 0; k < NUM_LABELS; k++)
            {
                giniIntermediate[j] += (labelCount[j][k]/valueCount[j]) * (labelCount[j][k]/valueCount[j]);
            }
            gini[j] = 1 - giniIntermediate[j];
        }
        weightedGini += valueCount[j]/totalCount*gini[j];
    }
    return weightedGini;
}

int countData(void)
{
     int count = 0;

    //Open input file
    FILE *inputFile = fopen("tennis.csv", "r");
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

int importData(int data[][NUM_ATTRIBUTES+1], int numInstances)
{
    FILE *inputFile = fopen("tennis.csv", "r");
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
                for (int j = 0; j < NUM_ATTRIBUTES+1; j++)
                {
                    data[i][j] = valueToInt(token);
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

//Convert value strings from input dataset to integers
int valueToInt(char* value)
{
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

int getMethod(void)
{
    char userInput;
    bool userInputValid;
    int method;
    do 
    {
        printf("Select attribute split method:\n\tInformation gain (I)\n\tMajority error (M)\n\tGini index (G)\n\n");
        scanf(" %c", &userInput);
        printf("\n");
        switch (userInput)
        {
            case 'I':
                userInputValid = true;
                method = 0;
                break;
            case 'M':
                userInputValid = true;
                method = 1;
                break;
            case 'G':
                userInputValid = true;
                method = 2;
                break;
            default:
                printf("Invalid selection\n\n");
                userInputValid = false;
        }
    } while (!userInputValid);
    return method;
}

int getMaxDepth(void)
{
    char userInput;
    bool userInputValid;
    int depth;

    do
    {
        printf("Select maximum tree depth (1-6)\n\n");
        scanf(" %c", &userInput);
        printf("\n");
        if (userInput >= '1' && userInput <= '6')
        {
            userInputValid = true;
            depth = userInput-48;
        }
        else
        {
            printf("Invalid selection\n\n");
            userInputValid = false;
        }
    } while (!userInputValid);
    return depth;
}

int getNextID(Branch tree[], int maxBranches)
{
    for (int i = 0; i < maxBranches; i++)
    {
        if (tree[i].active == false)
            return i;
    }
    return -1;
}

void printTree(Branch tree[], int maxBranches)
{
    int numBranches = 0;
    for (int i = 0; i < maxBranches; i++)
    {
        if (tree[i].active)
        {
            numBranches++;
        }
        else
        {
            break;
        }
    }

    for (int i = 0; i < numBranches; i++)
    {
        if (i == 0)
        {
            printf("Branch node 0 is the beginning of the tree and splits on attribute ");
            decodeAttribute(tree[i].attribute);
            printf("\n");
        }
        else
        {
            if (tree[i].label < 0)
            {
                printf("Branch node %d has value ", i);
                decodeValue(tree[tree[i].parent].attribute, tree[i].value);
                printf(" from parent branch %d and splits on attribute ", tree[i].parent);
                decodeAttribute(tree[i].attribute);
                printf("\n");
            }
            else
            {
                printf("Branch node %d has value ", i);
                decodeValue(tree[tree[i].parent].attribute, tree[i].value);
                printf(" from parent branch %d and label ", tree[i].parent);
                decodeLabel(tree[i].label);
                printf("\n");
            }
        }
    }
}

void decodeAttribute(int attribute)
{
    switch (attribute)
    {
        case 0:
            printf("'outlook'");
            break;
        case 1:
            printf("'temperature'");
            break;
        case 2:
            printf("'humidity'");
            break;
        case 3:
            printf("'wind'");
            break;
    }
}

void decodeValue(int attribute, int value)
{
    switch (attribute)
    {
        case 0:
            switch (value)
            {
                case 0:
                    printf("'sunny'");
                    break;
                case 1:
                    printf("'overcast'");
                    break;
                case 2:
                    printf("'rainy'");
                    break;
            }
            break;
        case 1:
            switch (value)
            {
                case 0:
                    printf("'hot'");
                    break;
                case 1:
                    printf("'medium'");
                    break;
                case 2:
                    printf("'cool'");
                    break;
            }
            break;
        case 2:
            switch (value)
            {
                case 0:
                    printf("'high'");
                    break;
                case 1:
                    printf("'normal'");
                    break;
                case 2:
                    printf("'low'");
                    break;
            }
            break;
        case 3:
            switch (value)
            {
                case 0:
                    printf("'strong'");
                    break;
                case 1:
                    printf("'weak'");
                    break;
            }
            break;
    }
}

void decodeLabel(int label)
{
    switch (label)
    {
        case 0:
            printf("'no'");
            break;
        case 1:
            printf("'yes'");
            break;
    }
}

void exportTree(Branch tree[], int maxBranches)
{
    printf("Enter name of .CSV file to export tree to:\n\n");
    char name[50];
    scanf("%s", name);
    strcat(name, ".csv");
    FILE *outputFile = fopen(name, "w");
    if (outputFile == NULL)
    {
        printf("Error opening file");
        return;
    }

    for (int i = 0; i < maxBranches; i++)
    {
        if (tree[i].active)
        {
            fprintf(outputFile, "%d,%d,%d,%d,%d", i, tree[i].attribute, tree[i].value, tree[i].label, tree[i].parent);
            for (int j = 0; j < MAX_VAL; j++)
            {
                fprintf(outputFile, ",%d", tree[i].leaf[j]);
            }
            fprintf(outputFile, "\n");
        }
        else
        {
            break;
        }
    }

    fclose(outputFile);
    return;
}