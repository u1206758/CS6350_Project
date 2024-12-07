#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <sys/time.h>

#define EPOCHS 10   //Number of epochs to repeat perceptron algorithm each run
#define MAX_VECTORS 500 //Size of weights array

#define NUM_TRAINING_INSTANCES 872   //872 instances in the training set
#define NUM_TESTING_INSTANCES 500   //500 instances in the test set
#define NUM_LABELS 2        //Binary label
#define NUM_ATTRIBUTES 4    //4 attributes


int import_training_data(float data[][NUM_ATTRIBUTES+1]);
int import_testing_data(float data[][NUM_ATTRIBUTES+1]);
void shuffle_data(float data[NUM_TRAINING_INSTANCES][NUM_ATTRIBUTES+1]);
void export_vectors(float w[4][MAX_VECTORS], float b[MAX_VECTORS], int count[MAX_VECTORS], int vectorIndex);

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

    float w[4][MAX_VECTORS];
    float b[MAX_VECTORS];
    int count[MAX_VECTORS];
    for (int i = 0; i < MAX_VECTORS; i++)
    {
        w[0][i] = 0;
        w[1][i] = 0;
        w[2][i] = 0;
        w[3][i] = 0;
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
            //Calculate prediction with current linear function
            prediction = w[0][vectorIndex]*training_data[i][0]+w[1][vectorIndex]*training_data[i][1]+w[2][vectorIndex]*training_data[i][2]+w[3][vectorIndex]*training_data[i][3]+b[vectorIndex];
            //Compare prediction to actual label
            if ((prediction <= 0 && training_data[i][4] == 1) || (prediction > 0 && training_data[i][4] == -1))
            {
                //Update function if necessary
                vectorIndex++;
                for (int j = 0; j < NUM_ATTRIBUTES; j++)
                {
                    w[j][vectorIndex] = w[j][vectorIndex-1] + training_data[i][4] * training_data[i][j];
                }
                b[vectorIndex] = b[vectorIndex-1] + training_data[i][4];
            }
            else
            {
                count[vectorIndex]++;
            }
        }
    }

    printf("\nLearning complete\n\n");
    for (int i = 0; i < vectorIndex+1; i++)
    {
        printf("%d: w=[%f %f %f %f], b=%f, count=%d\n",i,w[0][i],w[1][i],w[2][i],w[3][i],b[i],count[i]);
    }
    
    int errors = 0;
    float midPrediction[MAX_VECTORS];

    for (int i = 0; i < MAX_VECTORS; i++)
    {
        midPrediction[i] = 0;
    }

    //Make predictions on testing data using learned function
    for (int i = 0; i < NUM_TESTING_INSTANCES; i++)
    {
        prediction = 0;
        //Calculate prediction with current linear function
        for (int j = 0; j < vectorIndex+1; j++)
        {
            midPrediction[j] = w[0][j]*testing_data[i][0]+w[1][j]*testing_data[i][1]+w[2][j]*testing_data[i][2]+w[3][j]*testing_data[i][3]+b[j];
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
        if ((prediction <= 0 && testing_data[i][4] == 1) || (prediction > 0 && testing_data[i][4] == -1))
        {
            errors++;
        }
    }

    printf("\nPredicting complete\n");
    printf("\n%d incorrect predictions on %d instances\n", errors, NUM_TESTING_INSTANCES);
    printf("Prediction error: %.2f%% \n\n", ((float) errors / (float) NUM_TESTING_INSTANCES)*100);

    export_vectors(w, b, count, vectorIndex);

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

void export_vectors(float w[4][MAX_VECTORS], float b[MAX_VECTORS], int count[MAX_VECTORS], int vectorIndex)
{
    FILE *outputFile = fopen("voted_perceptron_vectors.csv", "w");
    if (outputFile == NULL)
    {
        printf("Error opening file");
        return;
    }

    fprintf(outputFile, "w[0],w[1],w[2],w[3],b,count\n");
    for (int i = 0; i < vectorIndex+1 ; i++)
    {
        fprintf(outputFile, "%f,%f,%f,%f,%f,%d", w[0][i], w[1][i], w[2][i], w[3][i], b[i], count[i]);
        fprintf(outputFile, "\n");
    }

    fclose(outputFile);
    return;
}