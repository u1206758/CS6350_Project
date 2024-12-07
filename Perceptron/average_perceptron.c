#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define EPOCHS 10   //Number of epochs to repeat perceptron algorithm each run

#define NUM_TRAINING_INSTANCES 872   //872 instances in the training set
#define NUM_TESTING_INSTANCES 500   //500 instances in the test set
#define NUM_LABELS 2        //Binary label
#define NUM_ATTRIBUTES 4    //4 attributes


int import_training_data(float data[][NUM_ATTRIBUTES+1]);
int import_testing_data(float data[][NUM_ATTRIBUTES+1]);
void shuffle_data(float data[NUM_TRAINING_INSTANCES][NUM_ATTRIBUTES+1]);

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

    float w[4] = {0, 0, 0, 0};
    float b = 0;
    float wAvg[4] = {0, 0, 0, 0};
    float bAvg = 0;
    int numVectors = 0;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        shuffle_data(training_data);
        for (int i = 0; i < NUM_TRAINING_INSTANCES; i++)
        {
            //Calculate prediction with current linear function
            float prediction = w[0]*training_data[i][0]+w[1]*training_data[i][1]+w[2]*training_data[i][2]+w[3]*training_data[i][3]+b;
            //Compare prediction to actual label
            if ((prediction <= 0 && training_data[i][4] == 1) || (prediction > 0 && training_data[i][4] == -1))
            {
                //Update function if necessary
                for (int j = 0; j < NUM_ATTRIBUTES; j++)
                {
                    w[j] += training_data[i][4] * training_data[i][j];
                    wAvg[j] += w[j];
                }
                b += training_data[i][4];
                bAvg += b;
                numVectors++;
            }
        }
    }

    wAvg[0] /= numVectors;
    wAvg[1] /= numVectors;
    wAvg[2] /= numVectors;
    wAvg[3] /= numVectors;
    bAvg /= numVectors;

    printf("\nLearning complete\n\n");
    printf("w[0]: %f\n", wAvg[0]);
    printf("w[1]: %f\n", wAvg[1]);
    printf("w[2]: %f\n", wAvg[2]);
    printf("w[3]: %f\n", wAvg[3]);
    printf("b: %f\n", bAvg);

    int errors = 0;

    //Make predictions on testing data using learned function
    for (int i = 0; i < NUM_TESTING_INSTANCES; i++)
    {
        //Calculate prediction with current linear function
        float prediction = wAvg[0]*testing_data[i][0]+wAvg[1]*testing_data[i][1]+wAvg[2]*testing_data[i][2]+wAvg[3]*testing_data[i][3]+bAvg;
        //Compare prediction to actual label
        if ((prediction <= 0 && testing_data[i][4] == 1) || (prediction > 0 && testing_data[i][4] == -1))
        {
            errors++;
        }
    }

    printf("\nPredicting complete\n");
    printf("\n%d incorrect predictions on %d instances\n", errors, NUM_TESTING_INSTANCES);
    printf("Prediction error: %.2f%% \n\n", ((float) errors / (float) NUM_TESTING_INSTANCES)*100);

    return 0;
}

int import_training_data(float data[][NUM_ATTRIBUTES+1])
{
    FILE *inputFile = fopen("train.csv", "r");
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
                    data[i][j] = atof(token);
                    if (j == 4 && data[i][j] == 0)
                    {
                        data[i][j] = -1;
                    }
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
    FILE *inputFile = fopen("test.csv", "r");
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
                for (int j = 0; j < NUM_ATTRIBUTES+1; j++)
                {
                    data[i][j] = atof(token);
                    if (j == 4 && data[i][j] == 0)
                    {
                        data[i][j] = -1;
                    }
                    token = strtok(NULL, ",\r\n");
                }
            }
        }
    }
    fclose(inputFile);
    return 0;
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
