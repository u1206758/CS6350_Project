#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define EPOCHS 10   //Number of epochs to repeat perceptron algorithm each run
#define MAX_VECTORS 60000 //Size of weights array

#define NUM_TRAINING_INSTANCES 25000   //872 instances in the training set
#define NUM_TESTING_INSTANCES 23842   //500 instances in the test set
#define NUM_LABELS 2        //Binary label
#define NUM_ATTRIBUTES 14    //4 attributes

bool isNumeric[NUM_ATTRIBUTES] = {true, false, true, false, true, false, false, false, false, false, true, true, true, false};

int import_training_data(float data[][NUM_ATTRIBUTES+1]);
int import_testing_data(float data[][NUM_ATTRIBUTES+1]);
float value_to_float(char* value, short attribute);
void shuffle_data(float data[NUM_TRAINING_INSTANCES][NUM_ATTRIBUTES+1]);
void export_submission(int myLabels[], int numInstances);

int main()
{
    float training_data[NUM_TRAINING_INSTANCES][NUM_ATTRIBUTES+1];
    float testing_data[NUM_TESTING_INSTANCES][NUM_ATTRIBUTES+1];

    //Import training data from CSV
    if (import_training_data(training_data) == -99)
        return 1;
    
    //Import testing data from CSV
    if (import_testing_data(testing_data) == -99)
        return 1;
    
    float w[NUM_ATTRIBUTES][MAX_VECTORS];
    float b[MAX_VECTORS];
    int count[MAX_VECTORS];
    for (int i = 0; i < MAX_VECTORS; i++)
    {
        for (int j = 0; j < NUM_ATTRIBUTES; j++)
        {
            w[j][i] = 0;
        }
        b[i] = 0;
        count[i] = 1;
    }
    int vectorIndex = 0;
    float prediction = 0;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        shuffle_data(training_data);
        for (int i = 0; i < NUM_TRAINING_INSTANCES; i++)
        {
            prediction = 0;
            //Calculate prediction with current linear function
            for (int j = 0; j < NUM_ATTRIBUTES; j++)
            {
                prediction += w[j][vectorIndex]*training_data[i][j];
            }
            prediction += b[vectorIndex];
            
            //Compare prediction to actual label
            if ((prediction <= 0 && training_data[i][NUM_ATTRIBUTES] == 1) || (prediction > 0 && training_data[i][NUM_ATTRIBUTES] == -1))
            {
                //Update function if necessary
                vectorIndex++;
                if (vectorIndex > MAX_VECTORS)
                {
                    printf("Out of vectors!   %d, %d\n",vectorIndex, i);
                    return 1;
                }
                for (int j = 0; j < NUM_ATTRIBUTES; j++)
                {
                    w[j][vectorIndex] = w[j][vectorIndex-1] + training_data[i][NUM_ATTRIBUTES] * training_data[i][j];
                }
                b[vectorIndex] = b[vectorIndex-1] + training_data[i][NUM_ATTRIBUTES];
            }
            else
            {
                count[vectorIndex]++;
            }
        }
    }

    printf("\nLearning complete\n\n");
    //for (int i = 0; i < NUM_ATTRIBUTES; i++)
    //{
    //    printf("w[%d]: %f\n", i, w[i]);
    //}
    //printf("b: %f\n", b);

    int errors = 0;
    float midPrediction[MAX_VECTORS];
    for (int i = 0; i < MAX_VECTORS; i++)
    {
        midPrediction[i] = 0;
    }

    //Make predictions on training data using learned function
    for (int i = 0; i < NUM_TRAINING_INSTANCES; i++)
    {
        prediction = 0;
        //Calculate prediction with current linear function
        for (int j = 0; j < vectorIndex; j++)
        {
            for (int k = 0; k < NUM_ATTRIBUTES; k++)
            {
                midPrediction[j] += w[k][j]*training_data[i][k];
            }
            midPrediction[j] += b[j];
            if (midPrediction[j] <= 0)
            {
                midPrediction[j] = -1;
            }
            else
            {
                midPrediction[j] = 1;
            }
            prediction += midPrediction[j] * count[j];
        }
        
        //Compare prediction to actual label
        if ((prediction <= 0 && training_data[i][NUM_ATTRIBUTES] == 1) || (prediction > 0 && training_data[i][NUM_ATTRIBUTES] == -1))
        {
            errors++;
        }
    }

    printf("\nTraining data predicting complete\n");
    printf("\n%d incorrect predictions on %d instances\n", errors, NUM_TRAINING_INSTANCES);
    printf("Prediction accuracy: %.2f%% \n\n", 100.0 - ((float) errors / (float) NUM_TRAINING_INSTANCES)*100);

    int myLabels[NUM_TRAINING_INSTANCES];

    for (int i = 0; i < MAX_VECTORS; i++)
    {
        midPrediction[i] = 0;
    }

    //Make predictions on testing data using learned function
    for (int i = 0; i < NUM_TESTING_INSTANCES; i++)
    {
        prediction = 0;
        //Calculate prediction with current linear function
        for (int j = 0; j < vectorIndex; j++)
        {
            for (int k = 0; k < NUM_ATTRIBUTES; k++)
            {
                midPrediction[j] += w[k][j]*testing_data[i][k];
            }
            midPrediction[j] += b[j];
            if (midPrediction[j] <= 0)
            {
                midPrediction[j] = -1;
            }
            else
            {
                midPrediction[j] = 1;
            }
            prediction += midPrediction[j] * count[j];
        }
        if (prediction < 0)
        {
            myLabels[i] = 0;
        }
        else
        {
            myLabels[i] = 1;
        }
    }
    
    //Export predictions to CSV
    printf("\nTesting data predicting complete\n");
    //export_submission(myLabels, NUM_TRAINING_INSTANCES);

    return 0;
}

int import_training_data(float data[][NUM_ATTRIBUTES+1])
{
    FILE *inputFile = fopen("train_final.csv", "r");
    if (inputFile == NULL)
    {
        printf("Error opening file");
        return -99;
    }

    char row[300];
    char *token;
    //Parse input CSV into data instance struct array
    while (feof(inputFile) != true)
    {
        for (int i = 0; i < NUM_TRAINING_INSTANCES; i++)
        {
            if (fgets(row, 300, inputFile) == NULL)
            {
                fclose(inputFile);
                return 0;
            }
            else
            {
                token = strtok(row, ",");
                for (int j = 0; j < NUM_ATTRIBUTES+1; j++)
                {
                    data[i][j] = value_to_float(token, j);
                    /*
                    data[i][j] = atof(token);
                    if (j == 4 && data[i][j] == 0)
                    {
                        data[i][j] = -1;
                    }*/
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

int import_testing_data(float data[][NUM_ATTRIBUTES+1])
{
    FILE *inputFile = fopen("test_final.csv", "r");
    if (inputFile == NULL)
    {
        printf("Error opening file");
        return -99;
    }

    char row[300];
    char *token;
    //Parse input CSV into data instance struct array
    while (feof(inputFile) != true)
    {
        for (int i = 0; i < NUM_TESTING_INSTANCES; i++)
        {
            if (fgets(row, 300, inputFile) == NULL)
            {
                fclose(inputFile);
                return 0;
            }
            else
            {
                token = strtok(row, ",");
                for (int j = 0; j < NUM_ATTRIBUTES; j++)
                {
                    data[i][j] = value_to_float(token, j);
                    /*
                    data[i][j] = atof(token);
                    if (j == 4 && data[i][j] == 0)
                    {
                        data[i][j] = -1;
                    }*/
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
}

//Convert value strings from input dataset to integers
float value_to_float(char* value, short attribute)
{
    if (isNumeric[attribute])
    {
        if (!(strcmp(value, "?")))
            return -1;
        else
        {
            return atof(value);
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
                    return -1;
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
                    return -1;
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
                    return -1;
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
                    return -1;
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
                    return -1;
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
                    return -1;
                break;
            case 9:
                if (!strcmp(value, "Female"))
                    return 0;
                if (!strcmp(value, "Male"))
                    return 1;
                if (!(strcmp(value, "?")))
                    return -1;
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
                    return -1;
                break;
            case 14:
                if (!strcmp(value, "0"))
                    return -1;
                if (!strcmp(value, "1"))
                    return 1;
                break;
        }
    }
    return -99;
}

//Fisher-Yates shuffling
void shuffle_data(float data[NUM_TRAINING_INSTANCES][NUM_ATTRIBUTES+1])
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    int usec = tv.tv_usec;
    srand48(usec);

    size_t i;
    for (i = NUM_TRAINING_INSTANCES - 1; i > 0; i--) {
        float temp[NUM_ATTRIBUTES+1];
        size_t j = (unsigned int) (drand48()*(i+1));
        for (int k = 0; k < NUM_ATTRIBUTES+1; k++)
        {
            temp[k] = data[j][k];
            data[j][k] = data[i][k];
            data[i][k] = temp[k];
        }
    }
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
        fprintf(outputFile, "%d,%d\n", i+1, myLabels[i]);
    }

    fclose(outputFile);
    return;
}