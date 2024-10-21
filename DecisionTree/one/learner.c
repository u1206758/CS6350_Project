//This program implements a decision tree learning algorithm for the salary evaluation task for the Kaggle competition.
//Users can select information gain, majority error, or gini index, and set a maximum depth.

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

#define NUM_I 25000 //25000 instances in the training set
#define NUM_LABELS 2 //Binary label, 0 = <=50k, 1= >50k
#define NUM_ATTRIBUTES 14 //14 attributes
/* Attributes & values
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
        14: 5th-6th
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
    FILE *inputFile = fopen("train_final.csv", "r");
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
    FILE *inputFile = fopen("train_final.csv", "r");
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
        if (atoi(value) <= thresholds[attribute])
        {
            //yes <=
            return 0;
        }
        else
        {
            //no >
            return 1;
        }
    }
    else
    {
        switch (attribute)
        {
            case 1:
                if (!strcmp(value, "Private"))
                    return 0;
                if (!strcmp(value, "Self-emp-not-inc"))
                    return 1;
                if (!strcmp(value, "Self-emp-inc"))
                    return 2;
                if (!strcmp(value, "Federal-gov"))
                    return 3;
                if (!strcmp(value, "Local-gov"))
                    return 4;
                if (!strcmp(value, "State-gov"))
                    return 5;
                if (!strcmp(value, "Without-pay"))
                    return 6;
                if (!strcmp(value, "Never-worked"))
                    return 7;
                break;
            case 3:
                if (!strcmp(value, "Bachelors"))
                    return 0;
                if (!strcmp(value, "Some-college"))
                    return 1;
                if (!strcmp(value, "11th"))
                    return 2;
                if (!strcmp(value, "HS-grad"))
                    return 3;
                if (!strcmp(value, "Prof-school"))
                    return 4;
                if (!strcmp(value, "Assoc-acdm"))
                    return 5;
                if (!strcmp(value, "Assoc-voc"))
                    return 6;
                if (!strcmp(value, "9th"))
                    return 7;
                if (!strcmp(value, "7th-8th"))
                    return 8;
                if (!strcmp(value, "12th"))
                    return 9;
                if (!strcmp(value, "Masters"))
                    return 10;
                if (!strcmp(value, "1st-4th"))
                    return 11;
                if (!strcmp(value, "10th"))
                    return 12;
                if (!strcmp(value, "Doctorate"))
                    return 13;
                if (!strcmp(value, "5th-6th"))
                    return 14;
                if (!strcmp(value, "Preschool"))
                    return 15;
                break;
            case 5:
                if (!strcmp(value, "Married-civ-spouse"))
                    return 0;
                if (!strcmp(value, "Divorced"))
                    return 1;
                if (!strcmp(value, "Never-married"))
                    return 2;
                if (!strcmp(value, "Separated"))
                    return 3;
                if (!strcmp(value, "Widowed"))
                    return 4;
                if (!strcmp(value, "Married-spouse-absent"))
                    return 5;
                if (!strcmp(value, "Married-AF-spouse"))
                    return 6;
                break;
            case 6:
                if (!strcmp(value, "Tech-support"))
                    return 0;
                if (!strcmp(value, "Craft-repair"))
                    return 1;
                if (!strcmp(value, "Other-service"))
                    return 2;
                if (!strcmp(value, "Sales"))
                    return 3;
                if (!strcmp(value, "Exec-managerial"))
                    return 4;
                if (!strcmp(value, "Prof-specialty"))
                    return 5;
                if (!strcmp(value, "Handlers-cleaners"))
                    return 6;
                if (!strcmp(value, "Machine-op-inspct"))
                    return 7;
                if (!strcmp(value, "Adm-clerical"))
                    return 8;
                if (!strcmp(value, "Farming-fishing"))
                    return 9;
                if (!strcmp(value, "Transport-moving"))
                    return 10;
                if (!strcmp(value, "Priv-house-serv"))
                    return 11;
                if (!strcmp(value, "Protective-serv"))
                    return 12;
                if (!strcmp(value, "Armed-Forces"))
                    return 13;
                break;
            case 7:
                if (!strcmp(value, "Wife"))
                    return 0;
                if (!strcmp(value, "Own-child"))
                    return 1;
                if (!strcmp(value, "Husband"))
                    return 2;
                if (!strcmp(value, "Not-in-family"))
                    return 3;
                if (!strcmp(value, "Other-relative"))
                    return 4;
                if (!strcmp(value, "Unmarried"))
                    return 5;
                break;
            case 8:
                if (!strcmp(value, "White"))
                    return 0;
                if (!strcmp(value, "Asian-Pac-Islander"))
                    return 1;
                if (!strcmp(value, "Amer-Indian-Eskimo"))
                    return 2;
                if (!strcmp(value, "Other"))
                    return 3;
                if (!strcmp(value, "Black"))
                    return 4;
                break;
            case 9:
                if (!strcmp(value, "Female"))
                    return 0;
                if (!strcmp(value, "Male"))
                    return 1;
                break;
            case 13:
                if (!strcmp(value, "United-States"))
                    return 0;
                if (!strcmp(value, "Cambodia"))
                    return 1;
                if (!strcmp(value, "England"))
                    return 2;
                if (!strcmp(value, "Puerto-Rico"))
                    return 3;
                if (!strcmp(value, "Canada"))
                    return 4;
                if (!strcmp(value, "Germany"))
                    return 5;
                if (!strcmp(value, "Outlying-US(Guam-USVI-etc)"))
                    return 6;
                if (!strcmp(value, "India"))
                    return 7;
                if (!strcmp(value, "Japan"))
                    return 8;
                if (!strcmp(value, "Greece"))
                    return 9;
                if (!strcmp(value, "South"))
                    return 10;
                if (!strcmp(value, "China"))
                    return 11;
                if (!strcmp(value, "Cuba"))
                    return 12;
                if (!strcmp(value, "Iran"))
                    return 13;
                if (!strcmp(value, "Honduras"))
                    return 14;
                if (!strcmp(value, "Philippines"))
                    return 15;
                if (!strcmp(value, "Italy"))
                    return 16;
                if (!strcmp(value, "Poland"))
                    return 17;
                if (!strcmp(value, "Jamaica"))
                    return 18;
                if (!strcmp(value, "Vietnam"))
                    return 19;
                if (!strcmp(value, "Mexico"))
                    return 20;
                if (!strcmp(value, "Portugal"))
                    return 21;
                if (!strcmp(value, "Ireland"))
                    return 22;
                if (!strcmp(value, "France"))
                    return 23;
                if (!strcmp(value, "Dominican-Republic"))
                    return 24;
                if (!strcmp(value, "Laos"))
                    return 25;
                if (!strcmp(value, "Ecuador"))
                    return 26;
                if (!strcmp(value, "Taiwan"))
                    return 27;
                if (!strcmp(value, "Haiti"))
                    return 28;
                if (!strcmp(value, "Columbia"))
                    return 29;
                if (!strcmp(value, "Hungary"))
                    return 30;
                if (!strcmp(value, "Guatemala"))
                    return 31;
                if (!strcmp(value, "Nicaragua"))
                    return 32;
                if (!strcmp(value, "Scotland"))
                    return 33;
                if (!strcmp(value, "Thailand"))
                    return 34;
                if (!strcmp(value, "Yugoslavia"))
                    return 35;
                if (!strcmp(value, "El-Salvador"))
                    return 36;
                if (!strcmp(value, "Trinadad&Tobago"))
                    return 37;
                if (!strcmp(value, "Peru"))
                    return 38;
                if (!strcmp(value, "Hong"))
                    return 39;
                if (!strcmp(value, "Holand-Netherlands"))
                    return 40;
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
        printf("Select maximum tree depth\n\n");
        scanf(" %s", userInput);
        printf("\n");
        //if (atoi(userInput) >= 1 && atoi(userInput) <= 16)
        //{
            userInputValid = true;
            depth = atoi(userInput);
        //}
        //else
        //{
            printf("Invalid selection\n\n");
        //    userInputValid = false;
        //}
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
            printf("'workclass'");
            break;
        case 2:
            printf("'fnlwgt'");
            break;
        case 3:
            printf("'education'");
            break;
        case 4:
            printf("'education-num'");
            break;
        case 5:
            printf("'marital-status'");
            break;
        case 6:
            printf("'occupation'");
            break;
        case 7:
            printf("'relationship'");
            break;
        case 8:
            printf("'race'");
            break;
        case 9:
            printf("'sex'");
            break;
        case 10:
            printf("'capital-gain'");
            break;
        case 11:
            printf("'capital-loss'");
            break;
        case 12:
            printf("'hours-per-week'");
            break;
        case 13:
            printf("'native-country'");
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
                    printf("'below median'");
                    break;
                case 1:
                    printf("'above median'");
                    break;
            }
            break;
        case 1:
            switch (value)
            {
                case 0:
                    printf("'Private'");
                    break;
                case 1:
                    printf("'Self-emp-not-inc'");
                    break;
                case 2:
                    printf("'Self-emp-inc'");
                    break;
                case 3:
                    printf("'Federal-gov'");
                    break;
                case 4:
                    printf("'Local-gov'");
                    break;
                case 5:
                    printf("'State-gov'");
                    break;
                case 6:
                    printf("'Without-pay'");
                    break;
                case 7:
                    printf("'Never-worked'");
                    break;
            }
            break;
        case 2:
            switch (value)
            {
                case 0:
                    printf("'below median'");
                    break;
                case 1:
                    printf("'above median'");
                    break;
            }
            break;
        case 3:
            switch (value)
            {
                case 0:
                    printf("'Bachelors'");
                    break;
                case 1:
                    printf("'Some-college'");
                    break;
                case 2:
                    printf("'11th'");
                    break;
                case 3:
                    printf("'HS-grad'");
                    break;
                case 4:
                    printf("'Prof-school'");
                    break;
                case 5:
                    printf("'Assoc-acdm'");
                    break;
                case 6:
                    printf("'Assoc-voc'");
                    break;
                case 7:
                    printf("'9th'");
                    break;
                case 8:
                    printf("'7th-8th'");
                    break;
                case 9:
                    printf("'12th'");
                    break;
                case 10:
                    printf("'Masters'");
                    break;
                case 11:
                    printf("'1st-4th'");
                    break;
                case 12:
                    printf("'10th'");
                    break;
                case 13:
                    printf("'Doctorate'");
                    break;
                case 14:
                    printf("'5th-6th'");
                    break;
                case 15:
                    printf("'Preschool'");
                    break;
            }
            break;
        case 4:
            switch (value)
            {             
                case 0:
                    printf("'below median'");
                    break;
                case 1:
                    printf("'above median'");
                    break;
            }
            break;
        case 5:
            switch (value)
            {
                case 0:
                    printf("Married-civ-spouse'");
                    break;
                case 1:
                    printf("'Divorced'");
                    break;
                case 2:
                    printf("'Never-married'");
                    break;
                case 3:
                    printf("'Separated'");
                    break;
                case 4:
                    printf("'Widowed'");
                    break;
                case 5:
                    printf("'Married-spouse-absent'");
                    break;
                case 6:
                    printf("'Married-AF-spouse'");
                    break;
            }
            break;
        case 6:
            switch (value)
            {
                case 0:
                    printf("'Tech-support'");
                    break;
                case 1:
                    printf("'Craft-repair'");
                    break;
                case 2:
                    printf("'Other-service'");
                    break;
                case 3:
                    printf("'Sales'");
                    break;
                case 4:
                    printf("'Exec-managerial'");
                    break;
                case 5:
                    printf("'Prof-specialty'");
                    break;
                case 6:
                    printf("'Handlers-cleaners'");
                    break;
                case 7:
                    printf("'Machine-op-inspct'");
                    break;
                case 8:
                    printf("'Adm-clerical'");
                    break;
                case 9:
                    printf("'Farming-fishing'");
                    break;
                case 10:
                    printf("'Transport-moving'");
                    break;
                case 11:
                    printf("'Priv-house-serv'");
                    break;
                case 12:
                    printf("'Protective-serv'");
                    break;
                case 13:
                    printf("'Armed-Forces'");
                    break;
            }
            break;
        case 7:
            switch (value)
            {
                case 0:
                    printf("'Wife'");
                    break;
                case 1:
                    printf("'Own-child'");
                    break;
                case 2:
                    printf("'Husband'");
                    break;
                case 3:
                    printf("'Not-in-family'");
                    break;
                case 4:
                    printf("'Other-relative'");
                    break;
                case 5:
                    printf("'Unmarried'");
                    break;
            }
            break;
        case 8:
            switch (value)
            {
                case 0:
                    printf("'White'");
                    break;
                case 1:
                    printf("'Asian-Pac-Islander'");
                    break;
                case 2:
                    printf("'Amer-Indian-Eskimo'");
                    break;
                case 3:
                    printf("'Other'");
                    break;
                case 4:
                    printf("'Black'");
                    break;
            }
            break;
        case 9:
            switch (value)
            {
                case 0:
                    printf("'Female'");
                    break;
                case 1:
                    printf("'Male'");
                    break;
            }
            break;
        case 10:
            switch (value)
            {
                case 0:
                    printf("'below median'");
                    break;
                case 1:
                    printf("'above median'");
                    break;
            }
            break;
        case 11:
            switch (value)
            {
                case 0:
                    printf("'below median'");
                    break;
                case 1:
                    printf("'above median'");
                    break;
            }
            break;
        case 12:
            switch (value)
            {
                case 0:
                    printf("'below median'");
                    break;
                case 1:
                    printf("'above median'");
                    break;
            }
            break;
        case 13:
            switch (value)
            {
                case 0:
                    printf("'United-States'");
                    break;
                case 1:
                    printf("'Cambodia'");
                    break;
                case 2:
                    printf("'England'");
                    break;
                case 3:
                    printf("'Puerto-Rico'");
                    break;
                case 4:
                    printf("'Canada'");
                    break;
                case 5:
                    printf("'Germany'");
                    break;
                case 6:
                    printf("'Outlying-US(Guam-USVI-etc)'");
                    break;
                case 7:
                    printf("'India'");
                    break;
                case 8:
                    printf("'Japan'");
                    break;
                case 9:
                    printf("'Greece'");
                    break;
                case 10:
                    printf("'South'");
                    break;
                case 11:
                    printf("'China'");
                    break;
                case 12:
                    printf("'Cuba'");
                    break;
                case 13:
                    printf("'Iran'");
                    break;
                case 14:
                    printf("'Honduras'");
                    break;
                case 15:
                    printf("'Philippines'");
                    break;
                case 16:
                    printf("'Italy'");
                    break;
                case 17:
                    printf("'Poland'");
                    break;
                case 18:
                    printf("'Jamaica'");
                    break;
                case 19:
                    printf("'Vietnam'");
                    break;
                case 20:
                    printf("'Mexico'");
                    break;
                case 21:
                    printf("'Portugal'");
                    break;
                case 22:
                    printf("'Ireland'");
                    break;
                case 23:
                    printf("'France'");
                    break;
                case 24:
                    printf("'Dominican-Republic'");
                    break;
                case 25:
                    printf("'Laos'");
                    break;
                case 26:
                    printf("'Ecuador'");
                    break;
                case 27:
                    printf("'Taiwan'");
                    break;
                case 28:
                    printf("'Haiti'");
                    break;
                case 29:
                    printf("'Columbia'");
                    break;
                case 30:
                    printf("'Hungary'");
                    break;
                case 31:
                    printf("'Guatemala'");
                    break;
                case 32:
                    printf("'Nicaragua'");
                    break;
                case 33:
                    printf("'Scotland'");
                    break;
                case 34:
                    printf("'Thailand'");
                    break;
                case 35:
                    printf("'Yugoslavia'");
                    break;
                case 36:
                    printf("'El-Salvador'");
                    break;
                case 37:
                    printf("'Trinadad&Tobago'");
                    break;
                case 38:
                    printf("'Peru'");
                    break;
                case 39:
                    printf("'Hong'");
                    break;
                case 40:
                    printf("'Holand-Netherlands'");
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
            printf("'<= 50k'");
            break;
        case 1:
            printf("'> 50k'");
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