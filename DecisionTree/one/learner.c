//This program implements a decision tree learning algorithm for the salary evaluation task for the Kaggle competition.
//Users can select information gain, majority error, or gini index, and set a maximum depth.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define NUM_I 25000 //25000 instances in the training set
#define NUM_LABELS 2 //Binary label
#define NUM_ATTRIBUTES 14 //14 attributes
/* 
    Attributes & values
    0: age
        continuous -> binary 0 = <= median threshold
    1: workclass
        0: Private
        1: Self-emp-not-inc
        2: Self-emp-inc
        3: Federal-gov
        4: Local-gov
        5: State-gov
        6: Without-pay
        7: Never-worked
    2: fnlwgt
        continuous -> binary 0 = <= median threshold
    3: education
        0: Bachelors
        1: Some-college
        2: 11th
        3: HS-grad
        4: Prof-school
        5: Assoc-acdm
        6: Assoc-voc
        7: 9th
        8: 7th-8th
        9: 12th
        10: Masters
        11: 1st-4th
        12: 10th
        13: Doctorate
        14: 5-th-6th
        15: Preschool
    4: education-num
        continuous -> binary 0 = <= median threshold
    5: marital-status
        0: Married-civ-spouse
        1: Divorced
        2: Never-married
        3: Separated
        4: Widowed
        5: Married-spouse-absent
        6: Married-AF-spouse
    6: occupation
        0: Tech-support
        1: Craft-repair
        2: Other-service
        3: Sales
        4: Exec-managerial
        5: Prof-specialty
        6: Handlers-cleaners
        7: Machine-op-inspct
        8: Adm-clerical
        9: Farming-fishing
        10: Transport-moving
        11: Priv-house-serv
        12: Protective-serv
        13: Armed-Forces
    7: relationship
        0: Wife
        1: Own-child
        2: Husband
        3: Not-in-family
        4: Other-relative
        5: Unmarried
    8: race
        0: White
        1: Asian-Pac-Islander
        2: Amer-Indian-Eskimo
        3: Other
        4: Black
    9: sex
        0: Female
        1: Male
    10: capital-gain
        continuous -> binary 0 = <= median threshold
    11: capital-loss
        continuous -> binary 0 = <= median threshold
    12: hours-per-week
        continuous -> binary 0 = <= median threshold
    13: native-country
        0: United-States
        1: Cambodia
        2: England
        3: Puerto-Rico
        4: Canada
        5: Germany
        6: Outlying-US(Guam-USVI-etc)
        7: India
        8: Japan
        9: Greece
        10: South
        11: China
        12: Cuba
        13: Iran
        14: Honduras
        15: Philippines
        16: Italy
        17: Poland
        18: Jamaica
        19: Vietnam
        20: Mexico
        21: Portugal
        22: Ireland
        23: France
        24: Dominican-Republic
        25: Laos
        26: Ecuador
        27: Taiwan
        28: Haiti
        29: Columbia
        30: Hungary
        31: Guatemala
        32: Nicaragua
        33: Scotland
        34: Thailand
        35: Yugoslavia
        36: El-Salvador
        37: Trinadad&Tobago
        38: Peru
        39: Hong
        40: Holand-Netherlands
*/
short numValues[NUM_ATTRIBUTES] = {2, 8, 2, 16, 2, 7, 14, 6, 5, 2, 2, 2, 2, 41};
//Precalculated thresholds (medians) of numerical attributes
float thresholds[NUM_ATTRIBUTES] = {};
bool isNumeric[NUM_ATTRIBUTES] = {true, false, true, false, true, false, false, false, false, false, true, true, true, false};
#define MAX_VAL 40 
#define MAX_BRANCH 5000

short splitLeaf(short currentInstances[NUM_I], short data[][NUM_ATTRIBUTES+1], short numInstances, short method, bool parentAttribute[NUM_ATTRIBUTES], short branchIndex);
float ig_initial(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances);
float ig_gain(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances, short attribute);
float me_initial(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances);
float me_gain(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances, short attribute);
float gini_initial(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances);
float gini_gain(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances, short attribute);

short countData(void);
short importData(short data[][NUM_ATTRIBUTES+1], short numInstances);
short valueToInt(char* value, short attribute);
short getMethod(void);
short getMaxDepth(void);

typedef struct
{
    bool active;    //Whether this branch slot is used by the tree
    short level;      //The depth in the tree this branch is on
    short parent;     //The ID of the parent branch
    short leaf[MAX_VAL];    //The IDs of the leaves of this branch
    short attribute;  //The attribute this branch is split on (if any)
    short value;      //The value a leaf represents when split from parent branch
    short label;      //The label for this branch (-99 no label, -2 all children labelled)
}Branch;

void printTree(Branch tree[], short maxBranches);
void decodeAttribute(short attribute);
void decodeValue(short attribute, short value);
void decodeLabel(short label);
void exportTree(Branch tree[], short maxBranches);

short getNextID(Branch tree[], short maxBranches);

int main()
{
    //Import data from CSV
    short numInstances = countData();
    if (numInstances == -99)
    {
        return 1;
    }
    short data[numInstances][NUM_ATTRIBUTES+1];
    importData(data, numInstances);
    short method = getMethod();
    short maxDepth = getMaxDepth();
    short maxBranches = MAX_BRANCH;
    Branch tree[maxBranches];
    short currentLevel = 1;
    bool allDone = false;

    //Initialize all possible branches
    for (short i = 0; i < maxBranches; i++)
    {
        tree[i].active = false;
        tree[i].level = -99;
        tree[i].parent = -99;
        for (short j = 0; j < MAX_VAL; j++)
            tree[i].leaf[j] = -99;
        tree[i].attribute = -99;
        tree[i].value = -99;
        tree[i].label = -99;
    }
    //Initialize head of tree
    //short currentInstances[maxBranches][numInstances];
    short (*currentInstances)[numInstances] = malloc(sizeof(*currentInstances) * maxBranches);

    for (short i = 0; i < maxBranches; i++)
    {
        for (short j = 0; j < numInstances; j++)
        {
            if (i == 0)
            {
                currentInstances[i][j] = j;
            }
            else
            {
                currentInstances[i][j] = -99;
            }
        }
    }
    short branchIndex = 0;
    tree[0].active = true;
    tree[0].level = 1;
    while(!allDone)
    {
       // printf("BI: %d, d: %d, a: %d\n", branchIndex, tree[branchIndex].level, tree[tree[branchIndex].parent].attribute);
        short lastLabel = -99;
        bool readyToLabel = true;
        bool allLeavesLabelled = true;
        bool hasLeaves = false;
        short labelCount[NUM_LABELS];
        for (short i = 0; i < NUM_LABELS; i++)
        {
            labelCount[i] = 0;
        }
        short maxLabelCount = 0;
        short maxLabel = -99;
        bool parentAttribute[NUM_ATTRIBUTES];
        short parentValue[NUM_ATTRIBUTES];
        for (short i = 0; i < NUM_ATTRIBUTES; i++)
        {
            parentAttribute[i] = false;
            parentValue[i] = -99;
        }

        //If current branch/leaf is at max level
        if (tree[branchIndex].level > maxDepth)
        {
            //find most common label for current value
            for (short i = 0; i < numInstances; i++)
            {
                if (data[i][tree[tree[branchIndex].parent].attribute] == tree[branchIndex].value)
                {
                    for (short j = 0; j < NUM_LABELS; j++)
                    {
                        if (data[i][NUM_ATTRIBUTES] == j)
                        {
                            labelCount[j]++;
                        }
                    }
                }
            }
            for (short i = 0; i < NUM_LABELS; i++)
            {
                if (labelCount[i] > maxLabelCount)
                {
                    maxLabelCount = labelCount[i];
                    maxLabel = i;
                }
            }
            //If branch value has no instances
            if (maxLabel == -99)
            {
                for (short i = 0; i < NUM_LABELS; i++)
                {
                    labelCount[i] = 0;
                }
                maxLabelCount = 0;
                //find most common label for current value
                for (short i = 0; i < numInstances; i++)
                {
                    if (data[i][tree[tree[tree[branchIndex].parent].parent].attribute] == tree[tree[branchIndex].parent].value)
                    {
                        for (short j = 0; j < NUM_LABELS; j++)
                        {
                            if (data[i][NUM_ATTRIBUTES] == j)
                            {
                                labelCount[j]++;
                            }
                        }
                    }
                }
                for (short i = 0; i < NUM_LABELS; i++)
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
            for (short j = 0; j < numValues[tree[tree[branchIndex].parent].attribute]; j++)
            {
                //If not, set flag false
                if (tree[tree[tree[branchIndex].parent].leaf[j]].label == -99)
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
            for (short i = 0; i < numInstances; i++)
            {   
                //If current instance is in current set
                if (currentInstances[branchIndex][i] != -99)
                {
                    //Check if all labels are the same
                    if (lastLabel == -99)
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
                if (lastLabel == -99)
                {
                    //find most common label for current value
                    for (short i = 0; i < numInstances; i++)
                    {
                        if (data[i][tree[tree[tree[branchIndex].parent].parent].attribute] == tree[tree[branchIndex].parent].value)
                        {
                            for (short j = 0; j < NUM_LABELS; j++)
                            {
                                if (data[i][NUM_ATTRIBUTES] == j)
                                {
                                    labelCount[j]++;
                                }
                            }
                        }
                    }
                    for (short i = 0; i < NUM_LABELS; i++)
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
                for (short j = 0; j < numValues[tree[branchIndex].attribute]; j++)
                {
                    //If not, set index to next unlabelled leaf
                    if (tree[tree[branchIndex].leaf[j]].label == -99)
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
                for (short i = 0; i < numValues[tree[branchIndex].attribute]; i++)
                {
                    if (tree[branchIndex].leaf[i] != -99)
                        {
                            hasLeaves = true;
                            break;
                        }
                }
                if (hasLeaves)
                {
                    //Check if all leaves are labelled
                    for (short i = 0; i < numValues[tree[branchIndex].attribute]; i++)
                    {
                        //if all are not labelled, set index to next unlabeleld
                        if (tree[tree[branchIndex].leaf[i]].label == -99)
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
                    short tempIndex = branchIndex;
                    while (tempIndex > -99)
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
                    for (short i = 0; i < numValues[tree[branchIndex].attribute]; i++)
                    {
                        tree[branchIndex].leaf[i] = getNextID(tree, maxBranches);
                        if (tree[branchIndex].leaf[i] == -99)
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
                        for (short j = 0; j < numInstances; j++)
                        {
                            //if the instance value mathing the leaf value for the parent attribute is also present in parent subset
                            if (data[j][tree[branchIndex].attribute] == i && currentInstances[branchIndex][j] != -99)
                            {
                                currentInstances[tree[branchIndex].leaf[i]][j] = j;
                            }
                            else
                            {
                                currentInstances[tree[branchIndex].leaf[i]][j] = -99;
                            }
                        }
                    }
                    //set branch index to first leaf
                    branchIndex = tree[branchIndex].leaf[0];
                }
            }
        }
    }
    char ans;
    printf("Print tree in terminal? (Y/N)\n\n");
    scanf(" %c", &ans);
    printf("\n");
    if (ans == 'Y')
    {
        printTree(tree, maxBranches);
    }
    exportTree(tree, maxBranches);
    return 0;
}

short splitLeaf(short currentInstances[NUM_I], short data[][NUM_ATTRIBUTES+1], short numInstances, short method, bool parentAttribute[NUM_ATTRIBUTES], short branchIndex)
{
    //Main algorithm loop
    float initialInformation;
    float attributeGain[NUM_ATTRIBUTES];
    float bestGain = -99;
    short bestAttribute = -99;
    
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
    for (short i = 0; i < NUM_ATTRIBUTES; i++)
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
    for (short i = 0; i < NUM_ATTRIBUTES; i++)
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
float ig_initial(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances)
{
    float totalCount = 0;
    float labelCount[NUM_LABELS];
    float init = 0;
    for (short i = 0; i < NUM_LABELS; i ++)
    {
        labelCount[i] = 0;
    }

    for (short i = 0; i < numInstances; i++)
    {
        if (subset[i] != -99)
        {
            totalCount++;
            for (short j = 0; j < NUM_LABELS; j++)
            {
                if (dataset[i][NUM_ATTRIBUTES] == j)
                {
                    labelCount[j]++;
                }       
            }
        }
    }

    for (short i = 0; i < NUM_LABELS; i++)
    {
        if (labelCount[i] != 0)
        {
            init -= ((labelCount[i]/totalCount)*log2(labelCount[i]/totalCount));
        }
    }

    return init;
}

//Calculate weighted entropy gain for each attribute in current instance set
float ig_gain(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances, short attribute)
{
    float totalCount = 0;
    float valueCount[numValues[attribute]];
    float labelCount[numValues[attribute]][NUM_LABELS];
    float entropy[numValues[attribute]];
    float weightedEntropy = 0;

    //Initialize values to zero
    for (short j = 0; j < numValues[attribute]; j++)
    {
        valueCount[j] = 0;
        for (short k = 0; k < NUM_LABELS; k++)
        {
            labelCount[j][k] = 0;
        }
        entropy[j] = 0;
    }

    //for each instance that is in the current subset
    for (short i = 0; i < numInstances; i++)
    {
        if (subset[i] != -99)
        {
            //count values for each attribute and their label
            for (short j = 0; j < numValues[attribute]; j++)
            {
                if (dataset[i][attribute] == j)
                {
                    valueCount[j]++;
                    totalCount++;
                    for (short k = 0; k < NUM_LABELS; k++)
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
    


    for (short j = 0; j < numValues[attribute]; j++)
    {
        for (short k = 0; k < NUM_LABELS; k++)
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
float me_initial(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances)
{
    float totalCount = 0;
    float labelCount[NUM_LABELS];
    for (short i = 0; i < NUM_LABELS; i++)
    {
        labelCount[i] = 0;
    }
    for (short i = 0; i < numInstances; i++)
    {
        if (subset[i] != -99)
        {
            totalCount++;
            for (short j = 0; j < NUM_LABELS; j++)
            {
                if (dataset[i][NUM_ATTRIBUTES] == j)
            {
                labelCount[j]++;
            }
            }
        }
    }

    float max = -99;

    for (short j = 0; j < NUM_LABELS; j++)
    {
        if (labelCount[j] > max)
        {
            max = labelCount[j];
        }
    }
    return 1 - (max/totalCount);
}

//Calculate weighted majority error gain for each attribute in current instance set
float me_gain(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances, short attribute)
{
    float totalCount = 0;
    float valueCount[numValues[attribute]];
    float labelCount[numValues[attribute]][NUM_LABELS];
    float me[numValues[attribute]];
    float weightedme = 0;

    //Initialize values to zero
    for (short j = 0; j < numValues[attribute]; j++)
    {
        valueCount[j] = 0;
        for (short k = 0; k < NUM_LABELS; k++)
        {
            labelCount[j][k] = 0;
        }
        me[j] = 0;
    }

    //for each instance that is in the current subset
    for (short i = 0; i < numInstances; i++)
    {
        if (subset[i] != -99)
        {
            //count values for each attribute and their label
            for (short j = 0; j < numValues[attribute]; j++)
            {
                if (dataset[i][attribute] == j)
                {
                    valueCount[j]++;
                    totalCount++;
                    for (short k = 0; k < NUM_LABELS; k++)
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
    for (short i = 0; i < numValues[attribute]; i++)
    {
        max[i] = -99;
    }

    for (short j = 0; j < numValues[attribute]; j++)
    {
        if (valueCount[j] == 0)
        {
            me[j] = 1;
        }
        else
        {
            for (short k = 0; k < NUM_LABELS; k++)
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
float gini_initial(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances)
{
    float totalCount = 0;
    float labelCount[NUM_LABELS];
    float gini = 0;
    for (short i = 0; i < NUM_LABELS; i++)
    {
        labelCount[i] = 0;
    }
    for (short i = 0; i < numInstances; i++)
    {
        if (subset[i] != -99)
        {
            totalCount++;
            for (short j = 0; j < NUM_LABELS; j++)
            {
                if (dataset[i][NUM_ATTRIBUTES] == j)
                {
                    labelCount[j]++;
                }
            }
        }
    }

    for (short i = 0; i < NUM_LABELS; i++)
    {
        gini += (labelCount[i]/totalCount) * (labelCount[i]/totalCount);
    }

    return 1 - gini;
}

//Calculate weighted gini gain for each attribute in current instance set
float gini_gain(short subset[], short dataset[][NUM_ATTRIBUTES+1], short numInstances, short attribute)
{
    float totalCount = 0;
    float valueCount[numValues[attribute]];
    float labelCount[numValues[attribute]][NUM_LABELS];
    float giniIntermediate[numValues[attribute]];
    float gini[numValues[attribute]];
    float weightedGini = 0;

    //Initialize values to zero
    for (short j = 0; j < numValues[attribute]; j++)
    {
        valueCount[j] = 0;
        for (short k = 0; k < NUM_LABELS; k++)
        {
            labelCount[j][k] = 0;
        }
        giniIntermediate[j] = 0;
        gini[j] = 0;
    }

    //for each instance that is in the current subset
    for (short i = 0; i < numInstances; i++)
    {
        if (subset[i] != -99)
        {
            //count values for each attribute and their label
            for (short j = 0; j < numValues[attribute]; j++)
            {

                if (dataset[i][attribute] == j)
                {
                    valueCount[j]++;
                    totalCount++;
                    for (short k = 0; k < NUM_LABELS; k++)
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
    


    for (short j = 0; j < numValues[attribute]; j++)
    {
        if (valueCount[j] == 0)
        {
            gini[j] = 1;
        }
        else
        {
            for (short k = 0; k < NUM_LABELS; k++)
            {
                giniIntermediate[j] += (labelCount[j][k]/valueCount[j]) * (labelCount[j][k]/valueCount[j]);
            }
            gini[j] = 1 - giniIntermediate[j];
        }
        weightedGini += valueCount[j]/totalCount*gini[j];
    }
    return weightedGini;
}

short countData(void)
{
     short count = 0;

    //Open input file
    FILE *inputFile = fopen("train.csv", "r");
    if (inputFile == NULL)
    {
        printf("Error opening file");
        return -99;
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

short importData(short data[][NUM_ATTRIBUTES+1], short numInstances)
{
    FILE *inputFile = fopen("train.csv", "r");
    if (inputFile == NULL)
    {
        printf("Error opening file");
        return -99;
    }

    char row[100];
    char *token;

    //Parse input CSV into data instance struct array
    while (feof(inputFile) != true)
    {
        for (short i = 0; i < numInstances; i++)
        {
            if (fgets(row, 100, inputFile) == NULL)
            {
                fclose(inputFile);
                return 0;
            }
            else
            {
                token = strtok(row, ",");
                for (short j = 0; j < NUM_ATTRIBUTES+1; j++)
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

//Convert value strings from input dataset to integers
short valueToInt(char* value, short attribute)
{
    if (isNumeric[attribute])
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
    return -99;
}

short getMethod(void)
{
    char userInput;
    bool userInputValid;
    short method;
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

short getMaxDepth(void)
{
    char userInput[50];
    bool userInputValid;
    short depth;

    do
    {
        printf("Select maximum tree depth (1-16)\n\n");
        scanf(" %s", userInput);
        printf("\n");
        if (atoi(userInput) >= 1 && atoi(userInput) <= 16)
        {
            userInputValid = true;
            depth = atoi(userInput);
        }
        else
        {
            printf("Invalid selection\n\n");
            userInputValid = false;
        }
    } while (!userInputValid);
    return depth;
}

short getNextID(Branch tree[], short maxBranches)
{
    for (short i = 0; i < maxBranches; i++)
    {
        if (tree[i].active == false)
            return i;
    }
    return -99;
}

void printTree(Branch tree[], short maxBranches)
{
    short numBranches = 0;
    for (short i = 0; i < maxBranches; i++)
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

    for (short i = 0; i < numBranches; i++)
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

void decodeAttribute(short attribute)
{
    switch (attribute)
    {
        case 0:
            printf("'age'");
            break;
        case 1:
            printf("'job'");
            break;
        case 2:
            printf("'marital'");
            break;
        case 3:
            printf("'education'");
            break;
        case 4:
            printf("'default'");
            break;
        case 5:
            printf("'balance'");
            break;
        case 6:
            printf("'housing'");
            break;
        case 7:
            printf("'loan'");
            break;
        case 8:
            printf("'contact'");
            break;
        case 9:
            printf("'day'");
            break;
        case 10:
            printf("'month'");
            break;
        case 11:
            printf("'duration'");
            break;
        case 12:
            printf("'campaign'");
            break;
        case 13:
            printf("'pdays'");
            break;
        case 14:
            printf("'previous'");
            break;
        case 15:
            printf("'poutcome'");
            break;
    }
}

void decodeValue(short attribute, short value)
{
    switch (attribute)
    {
        case 0:
            switch (value)
            {
                case 0:
                    printf("'above mean'");
                    break;
                case 1:
                    printf("'below mean'");
                    break;
            }
            break;
        case 1:
            switch (value)
            {
                case 0:
                    printf("'admin'");
                    break;
                case 1:
                    printf("'unknown'");
                    break;
                case 2:
                    printf("'unemployed'");
                    break;
                case 3:
                    printf("'management'");
                    break;
                case 4:
                    printf("'housemaid'");
                    break;
                case 5:
                    printf("'entrepreneur'");
                    break;
                case 6:
                    printf("'student'");
                    break;
                case 7:
                    printf("'blue-collar'");
                    break;
                case 8:
                    printf("'self-employed'");
                    break;
                case 9:
                    printf("'retired'");
                    break;
                case 10:
                    printf("'technician'");
                    break;
                case 11:
                    printf("'services'");
                    break;
            }
            break;
        case 2:
            switch (value)
            {
                case 0:
                    printf("'married'");
                    break;
                case 1:
                    printf("'divorced'");
                    break;
                case 2:
                    printf("'single'");
                    break;
            }
            break;
        case 3:
            switch (value)
            {
                case 0:
                    printf("'unknown'");
                    break;
                case 1:
                    printf("'secondary'");
                    break;
                case 3:
                    printf("'primary'");
                    break;
                case 4:
                    printf("'tertiary'");
                    break;
            }
            break;
        case 4:
            switch (value)
            {
                case 0:
                    printf("'yes'");
                    break;
                case 1:
                    printf("'no'");
                    break;
            }
            break;
        case 5:
            switch (value)
            {
                case 0:
                    printf("'above median'");
                    break;
                case 1:
                    printf("'below median'");
                    break;
            }
            break;
        case 6:
            switch (value)
            {
                case 0:
                    printf("'yes'");
                    break;
                case 1:
                    printf("'no'");
                    break;
            }
            break;
        case 7:
            switch (value)
            {
                case 0:
                    printf("'yes'");
                    break;
                case 1:
                    printf("'no'");
                    break;
            }
            break;
        case 8:
            switch (value)
            {
                case 0:
                    printf("'unknown'");
                    break;
                case 1:
                    printf("'telephone'");
                    break;
                case 2:
                    printf("'cellular'");
                    break;
            }
            break;
        case 9:
            switch (value)
            {
                case 0:
                    printf("'above median'");
                    break;
                case 1:
                    printf("'below median'");
                    break;
            }
            break;
        case 10:
            switch (value)
            {
                case 0:
                    printf("'jan'");
                    break;
                case 1:
                    printf("'feb'");
                    break;
                case 2:
                    printf("'mar'");
                    break;
                case 3:
                    printf("'apr'");
                    break;
                case 4:
                    printf("'may'");
                    break;
                case 5:
                    printf("'jun'");
                    break;
                case 6:
                    printf("'jul'");
                    break;
                case 7:
                    printf("'aug'");
                    break;
                case 8:
                    printf("'sep'");
                    break;
                case 9:
                    printf("'oct'");
                    break;
                case 10:
                    printf("'nov'");
                    break;
                case 11:
                    printf("'dec'");
                    break;
            }
            break;
        case 11:
            switch (value)
            {
                case 0:
                    printf("'above median'");
                    break;
                case 1:
                    printf("'below median'");
                    break;
            }
            break;
        case 12:
            switch (value)
            {
                case 0:
                    printf("'above median'");
                    break;
                case 1:
                    printf("'below median'");
                    break;
            }
            break;
        case 13:
            switch (value)
            {
                case 0:
                    printf("'above median'");
                    break;
                case 1:
                    printf("'below median'");
                    break;
            }
            break;
        case 14:
            switch (value)
            {
                case 0:
                    printf("'above median'");
                    break;
                case 1:
                    printf("'below median'");
                    break;
            }
            break;
        case 15:
            switch (value)
            {
                case 0:
                    printf("'unknown'");
                    break;
                case 1:
                    printf("'other'");
                    break;
                case 2:
                    printf("'failure'");
                    break;
                case 3:
                    printf("success");
                    break;
            }
            break;
    }
}

void decodeLabel(short label)
{
    switch (label)
    {
        case 0:
            printf("'yes'");
            break;
        case 1:
            printf("'no'");
            break;
    }
}

void exportTree(Branch tree[], short maxBranches)
{
    printf("\nEnter name of file to export tree to:\n\n");
    char name[50];
    scanf("%s", name);
    FILE *outputFile = fopen(name, "w");
    if (outputFile == NULL)
    {
        printf("Error opening file");
        return;
    }

    for (short i = 0; i < maxBranches; i++)
    {
        if (tree[i].active)
        {
            fprintf(outputFile, "%d,%d,%d,%d,%d", i, tree[i].attribute, tree[i].value, tree[i].label, tree[i].parent);
            for (short j = 0; j < MAX_VAL; j++)
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