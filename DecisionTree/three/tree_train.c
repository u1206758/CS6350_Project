#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define NUM_ATTRIBUTES 14
#define MAX_VAL 41
#define TREE_VAL 46
int numValues[NUM_ATTRIBUTES] = {2, 8, 2, 16, 2, 7, 14, 6, 5, 2, 2, 2, 2, 41};
float thresholds[NUM_ATTRIBUTES] = {37, 0, 177299.5, 0, 10, 0, 0, 0, 0, 0, 0, 0, 40, 0};
bool isNumeric[NUM_ATTRIBUTES] = {true, false, true, false, true, false, false, false, false, false, true, true, true, false};

int count_entries(char fileName[]);
int import_data(char fileName[], int data[][NUM_ATTRIBUTES+1], int numInstances, int numAttributes, int dataID[numInstances]);
int import_tree(char fileName[], int tree[][TREE_VAL], int numInstances, int numAttributes);
int value_to_int(char* value, int attribute);
void export_submission(int myLabels[], int numInstances);
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
    int numInstances = count_entries(userInput);
    if (numInstances == -99)
    {
        return 1;
    }
    int data[numInstances][NUM_ATTRIBUTES+1];
    int dataID[numInstances];
    import_data(userInput, data, numInstances, NUM_ATTRIBUTES, dataID);
    //Find most common value of each attribute and replace unkown with it in dataset
    int commonVal[NUM_ATTRIBUTES];
    int valCount[NUM_ATTRIBUTES][MAX_VAL];
    //Initialize counts to 0
    for (int i = 0; i < NUM_ATTRIBUTES; i++)
    {
        commonVal[i] = -1;
        for (int j = 0; j < MAX_VAL; j++)
        {
            valCount[i][j] = 0;
        }   
    }
    //Count each occurence of each value in the dataset
    for (int i = 0; i < numInstances; i++)
    {
        for (int j = 0; j < NUM_ATTRIBUTES; j++)
        {
            for (int k = 0; k < numValues[j]; k++)
            {
                if (data[i][j] == k)
                {
                   valCount[j][k]++;
                }
            }
        }
    }
    //Find the most common value of each attribute
    for (int i = 0; i < NUM_ATTRIBUTES; i++)
    {
        for (int j = 0; j < numValues[i]; j++)
        {
            if (valCount[i][j] > valCount[i][commonVal[i]])
            {
                commonVal[i] = j;
            }
        }
    }
    //Replace unkown values with most common
    for (int i = 0; i < numInstances; i++)
    {
        for (int j = 0; j < NUM_ATTRIBUTES; j++)
        {
            if (data[i][j] == numValues[j])
            {
                data[i][j] = commonVal[j];
            }
        }
    }
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
    int numBranches = count_entries(userInput);
    if (numBranches == -99)
    {
        return 1;
    }
    int tempTree[numBranches][TREE_VAL];
    //Branch tree[numBranches];
    Branch* tree = (Branch*)malloc(numBranches * sizeof(Branch));
    import_tree(userInput, tempTree, numBranches, TREE_VAL);
    for (int i = 0; i < numBranches; i++)
    {
        tree[i].id = tempTree[i][0];
        tree[i].attribute = tempTree[i][1];
        tree[i].value = tempTree[i][2];
        tree[i].label = tempTree[i][3];
        tree[i].parent = tempTree[i][4];
        for (int j = 0; j < numValues[tree[i].attribute]; j++)
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
        //printf("ii: %d\n", instanceIndex);
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
            for (int j = 0; j < numValues[tree[branchIndex].attribute]; j++)
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
    
    export_submission(myLabels, numInstances);
}

int count_entries(char fileName[])
{
     int count = 0;

    //Open input file
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file: %s\n", fileName);
        return -99;
    }

    char row[300];

    //Count number of instances in input file
    while (feof(inputFile) != true)
    {
        if (fgets(row, 300, inputFile) == NULL)
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

int import_data(char fileName[], int data[][NUM_ATTRIBUTES+1], int numInstances, int numAttributes, int dataID[numInstances])
{
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file: %s\n", fileName);
        return -99;
    }

    char row[300];
    char *token;

    //Parse input CSV into data instance struct array
    while (feof(inputFile) != true)
    {
        for (int i = 0; i < numInstances; i++)
        {
            if (fgets(row, 300, inputFile) == NULL)
            {
                fclose(inputFile);
                return 0;
            }
            else
            {
                token = strtok(row, ",");
                for (int j = 0; j < numAttributes+1; j++)
                {
                    //if (j == 0)
                    //{
                    //    dataID[i] = atoi(token);
                    //}
                    //else
                    //{
                        //data[i][j-1] = value_to_int(token, j-1);
                        dataID[i]=i;
                        data[i][j] = value_to_int(token, j);
                        token = strtok(NULL, ",\r\n");
                    //}
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

int import_tree(char fileName[], int tree[][TREE_VAL], int numInstances, int numAttributes)
{
    FILE *inputFile = fopen(fileName, "r");
    if (inputFile == NULL)
    {
        printf("Error opening file: %s\n", fileName);
        return -99;
    }

    char row[300];
    char *token;

    //Parse input CSV into data instance struct array
    while (feof(inputFile) != true)
    {
        for (int i = 0; i < numInstances; i++)
        {
            if (fgets(row, 300, inputFile) == NULL)
            {
                fclose(inputFile);
                return 0;
            }
            else
            {
                token = strtok(row, ",");
                for (int j = 0; j < numAttributes; j++)
                {
                    tree[i][j] = value_to_int(token, -2);
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

//Convert value strings from input dataset to integers
int value_to_int(char* value, int attribute)
{
    if (attribute == -2)
    {
        return atoi(value);
    }
    else if (isNumeric[attribute])
    {
        if (!(strcmp(value, "?")))
            return 2;
        else
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
                if (!(strcmp(value, "?")))
                    return 8;
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
                if (!(strcmp(value, "?")))
                    return 16;
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
                if (!(strcmp(value, "?")))
                    return 7;
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
                if (!(strcmp(value, "?")))
                    return 14;
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
                if (!(strcmp(value, "?")))
                    return 6;
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
                if (!(strcmp(value, "?")))
                    return 5;
                break;
            case 9:
                if (!strcmp(value, "Female"))
                    return 0;
                if (!strcmp(value, "Male"))
                    return 1;
                if (!(strcmp(value, "?")))
                    return 2;
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
                if (!(strcmp(value, "?")))
                    return 41;
                break;
            case 14:
                if (!strcmp(value, "0"))
                    return 0;
                if (!strcmp(value, "1"))
                    return 1;
                break;
        }
    }
    return -99;
}

void export_submission(int myLabels[], int numInstances)
{
    printf("\nEnter name of file to export submission to:\n\n");
    char name[50];
    scanf("%s", name);
    FILE *outputFile = fopen(name, "w");
    if (outputFile == NULL)
    {
        printf("Error opening file");
        return;
    }

    fprintf(outputFile, "ID,Prediction\n");
    
    for (short i = 0; i < numInstances; i++)
    {
        fprintf(outputFile, "%d,%d\n", i, myLabels[i]);
    }

    fclose(outputFile);
    return;
}