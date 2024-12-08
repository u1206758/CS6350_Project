#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define EPOCHS 100   //Number of epochs to repeat perceptron algorithm each run

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
    
    float w[NUM_ATTRIBUTES] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float b = 0;
    float prediction = 0;

    int updates = 0;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        shuffle_data(training_data);
        for (int i = 0; i < NUM_TRAINING_INSTANCES; i++)
        {
            prediction = 0;
            //Calculate prediction with current linear function
            for (int j = 0; j < NUM_ATTRIBUTES; j++)
            {
                prediction += w[j]*training_data[i][j];
            }
            prediction += b;
            
            //Compare prediction to actual label
            if ((prediction <= 0 && training_data[i][NUM_ATTRIBUTES] == 1) || (prediction > 0 && training_data[i][NUM_ATTRIBUTES] == -1))
            {
                //Update function if necessary
                updates++;
                for (int j = 0; j < NUM_ATTRIBUTES; j++)
                {
                    w[j] += training_data[i][NUM_ATTRIBUTES] * training_data[i][j];
                }
                b += training_data[i][NUM_ATTRIBUTES];
            }
        }
    }

    printf("\nLearning complete\n\n");
    printf("%d updates\n",updates);
    for (int i = 0; i < NUM_ATTRIBUTES; i++)
    {
        printf("w[%d]: %f\n", i, w[i]);
    }
    printf("b: %f\n", b);

    int errors = 0;

    //Make predictions on training data using learned function
    for (int i = 0; i < NUM_TRAINING_INSTANCES; i++)
    {
        prediction = 0;
        //Calculate prediction with current linear function
        for (int j = 0; j < NUM_ATTRIBUTES; j++)
        {
            prediction += w[j]*training_data[i][j];
        }
        prediction += b;
        
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

    //Make predictions on testing data using learned function
    for (int i = 0; i < NUM_TESTING_INSTANCES; i++)
    {
        prediction = 0;
        //Calculate prediction with current linear function
        for (int j = 0; j < NUM_ATTRIBUTES; j++)
        {
            prediction += w[j]*testing_data[i][j];
        }
        prediction += b;
        
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
                    if (j < NUM_ATTRIBUTES && !isNumeric[j])
                    {
                        data[i][j] *= 100;
                    }
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
                    return -56;
                if (!strcmp(value, "Self-emp-not-inc"))
                    return -44;
                if (!strcmp(value, "Self-emp-inc"))
                    return 13;
                if (!strcmp(value, "Federal-gov"))
                    return -23;
                if (!strcmp(value, "Local-gov"))
                    return -42;
                if (!strcmp(value, "State-gov"))
                    return -47;
                if (!strcmp(value, "Without-pay"))
                    return -87;
                if (!strcmp(value, "Never-worked"))
                    return -100;
                if (!(strcmp(value, "?")))
                    return -81;
                break;
            case 3:
                if (!strcmp(value, "Bachelors"))
                    return -19;
                if (!strcmp(value, "Some-college"))
                    return -61;
                if (!strcmp(value, "11th"))
                    return -91;
                if (!strcmp(value, "HS-grad"))
                    return -68;
                if (!strcmp(value, "Prof-school"))
                    return 51;
                if (!strcmp(value, "Assoc-acdm"))
                    return -47;
                if (!strcmp(value, "Assoc-voc"))
                    return -51;
                if (!strcmp(value, "9th"))
                    return -89;
                if (!strcmp(value, "7th-8th"))
                    return -87;
                if (!strcmp(value, "12th"))
                    return -85;
                if (!strcmp(value, "Masters"))
                    return 11;
                if (!strcmp(value, "1st-4th"))
                    return -96;
                if (!strcmp(value, "10th"))
                    return -87;
                if (!strcmp(value, "Doctorate"))
                    return 54;
                if (!strcmp(value, "5th-6th"))
                    return -90;
                if (!strcmp(value, "Preschool"))
                    return -95;
                if (!(strcmp(value, "?")))
                    return 0;
                break;
            case 5:
                if (!strcmp(value, "Married-civ-spouse"))
                    return -11;
                if (!strcmp(value, "Divorced"))
                    return 0;
                if (!strcmp(value, "Never-married"))
                    return -90;
                if (!strcmp(value, "Separated"))
                    return -83;
                if (!strcmp(value, "Widowed"))
                    return -82;
                if (!strcmp(value, "Married-spouse-absent"))
                    return -82;
                if (!strcmp(value, "Married-AF-spouse"))
                    return -26;
                if (!(strcmp(value, "?")))
                    return 0;
                break;
            case 6:
                if (!strcmp(value, "Tech-support"))
                    return -41;
                if (!strcmp(value, "Craft-repair"))
                    return -54;
                if (!strcmp(value, "Other-service"))
                    return -92;
                if (!strcmp(value, "Sales"))
                    return -46;
                if (!strcmp(value, "Exec-managerial"))
                    return -6;
                if (!strcmp(value, "Prof-specialty"))
                    return -10;
                if (!strcmp(value, "Handlers-cleaners"))
                    return -87;
                if (!strcmp(value, "Machine-op-inspct"))
                    return 0;
                if (!strcmp(value, "Adm-clerical"))
                    return -72;
                if (!strcmp(value, "Farming-fishing"))
                    return -77;
                if (!strcmp(value, "Transport-moving"))
                    return -59;
                if (!strcmp(value, "Priv-house-serv"))
                    return -98;
                if (!strcmp(value, "Protective-serv"))
                    return -38;
                if (!strcmp(value, "Armed-Forces"))
                    return -40;
                if (!(strcmp(value, "?")))
                    return -81;
                break;
            case 7:
                if (!strcmp(value, "Wife"))
                    return -5;
                if (!strcmp(value, "Own-child"))
                    return -97;
                if (!strcmp(value, "Husband"))
                    return -11;
                if (!strcmp(value, "Not-in-family"))
                    return -79;
                if (!strcmp(value, "Other-relative"))
                    return -92;
                if (!strcmp(value, "Unmarried"))
                    return -87;
                if (!(strcmp(value, "?")))
                    return 0;
                break;
            case 8:
                if (!strcmp(value, "White"))
                    return -49;
                if (!strcmp(value, "Asian-Pac-Islander"))
                    return -47;
                if (!strcmp(value, "Amer-Indian-Eskimo"))
                    return -76;
                if (!strcmp(value, "Other"))
                    return -71;
                if (!strcmp(value, "Black"))
                    return -77;
                if (!(strcmp(value, "?")))
                    return 0;
                break;
            case 9:
                if (!strcmp(value, "Female"))
                    return -78;
                if (!strcmp(value, "Male"))
                    return -39;
                if (!(strcmp(value, "?")))
                    return 0;
                break;
            case 13:
                if (!strcmp(value, "United-States"))
                    return -51;
                if (!strcmp(value, "Cambodia"))
                    return -57;
                if (!strcmp(value, "England"))
                    return -24;
                if (!strcmp(value, "Puerto-Rico"))
                    return -82;
                if (!strcmp(value, "Canada"))
                    return -38;
                if (!strcmp(value, "Germany"))
                    return -37;
                if (!strcmp(value, "Outlying-US(Guam-USVI-etc)"))
                    return -83;
                if (!strcmp(value, "India"))
                    return -18;
                if (!strcmp(value, "Japan"))
                    return -23;
                if (!strcmp(value, "Greece"))
                    return -40;
                if (!strcmp(value, "South"))
                    return -70;
                if (!strcmp(value, "China"))
                    return 0;
                if (!strcmp(value, "Cuba"))
                    return -51;
                if (!strcmp(value, "Iran"))
                    return -14;
                if (!strcmp(value, "Honduras"))
                    return -100;
                if (!strcmp(value, "Philippines"))
                    return -43;
                if (!strcmp(value, "Italy"))
                    return -38;
                if (!strcmp(value, "Poland"))
                    return -60;
                if (!strcmp(value, "Jamaica"))
                    return -70;
                if (!strcmp(value, "Vietnam"))
                    return -77;
                if (!strcmp(value, "Mexico"))
                    return -91;
                if (!strcmp(value, "Portugal"))
                    return -67;
                if (!strcmp(value, "Ireland"))
                    return -14;
                if (!strcmp(value, "France"))
                    return -37;
                if (!strcmp(value, "Dominican-Republic"))
                    return -87;
                if (!strcmp(value, "Laos"))
                    return -86;
                if (!strcmp(value, "Ecuador"))
                    return -78;
                if (!strcmp(value, "Taiwan"))
                    return -15;
                if (!strcmp(value, "Haiti"))
                    return -73;
                if (!strcmp(value, "Columbia"))
                    return -85;
                if (!strcmp(value, "Hungary"))
                    return 0;
                if (!strcmp(value, "Guatemala"))
                    return -91;
                if (!strcmp(value, "Nicaragua"))
                    return -93;
                if (!strcmp(value, "Scotland"))
                    return -71;
                if (!strcmp(value, "Thailand"))
                    return -43;
                if (!strcmp(value, "Yugoslavia"))
                    return -14;
                if (!strcmp(value, "El-Salvador"))
                    return -84;
                if (!strcmp(value, "Trinadad&Tobago"))
                    return 0;
                if (!strcmp(value, "Peru"))
                    return -74;
                if (!strcmp(value, "Hong"))
                    return -58;
                if (!strcmp(value, "Holand-Netherlands"))
                    return -0;
                if (!(strcmp(value, "?")))
                    return -50;
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