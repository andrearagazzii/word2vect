#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// TODO: function to feed data

#define MAX_WORD_SIZE 20
#define MAX_WORD_NUMBER 100

#define WINDOW 4
#define HIDDEN_LAYER_SIZE 5
#define EPOCHS 100
#define LEARNING_RATE 0.1

typedef struct {
	char word[MAX_WORD_SIZE];
	float vect[MAX_WORD_NUMBER];
} VocItem;

typedef struct {
	int dim;
	VocItem voc[MAX_WORD_NUMBER];
} Vocabulary;

void createVocabulary(char file_path[], Vocabulary *v);
int tokenize(char tokens[][MAX_WORD_SIZE], char str[]);
VocItem *encode(Vocabulary *v, char word[]);
float softmax(float input[], float output[], int dim);
float random_float();
float distance(float vectA[], float vectB[], int dim);

int main() {
	Vocabulary v;

	// create the vocabulary
	createVocabulary("voc.txt", &v);

	char str1[] = "he is a king";
	char str2[] = "she is a queen";
	char str3[] = "he is a man";
	char str4[] = "she is a woman";

	char train_data[50][2][MAX_WORD_SIZE];

	char tokens[MAX_WORD_NUMBER][MAX_WORD_SIZE];
	int token_count;
	int sample_size = 0;

	// create training data
	token_count = tokenize(tokens, str1);
	for (int i = 0; i < token_count - 1; i++) {
		for (int j = i+1; j < token_count && j < i + 1 + WINDOW; j++) {
			strcpy(train_data[sample_size][0], tokens[i]);
			strcpy(train_data[sample_size][1], tokens[j]);
			sample_size++;
		}
	}

	token_count = tokenize(tokens, str2);
	for (int i = 0; i < token_count - 1; i++) {
		for (int j = i+1; j < token_count && j < i + 1 + WINDOW; j++) {
			strcpy(train_data[sample_size][0], tokens[i]);
			strcpy(train_data[sample_size][1], tokens[j]);
			sample_size++;
		}
	}
	
	token_count = tokenize(tokens, str3);
	for (int i = 0; i < token_count - 1; i++) {
		for (int j = i+1; j < token_count && j < i + 1 + WINDOW; j++) {
			strcpy(train_data[sample_size][0], tokens[i]);
			strcpy(train_data[sample_size][1], tokens[j]);
			sample_size++;
		}
	}
	
	token_count = tokenize(tokens, str4);
	for (int i = 0; i < token_count - 1; i++) {
		for (int j = i+1; j < token_count && j < i + 1 + WINDOW; j++) {
			strcpy(train_data[sample_size][0], tokens[i]);
			strcpy(train_data[sample_size][1], tokens[j]);
			sample_size++;
		}
	}


	// TODO: create the nn for embeddings
	
	float w_h[v.dim][HIDDEN_LAYER_SIZE];
	float b_h[HIDDEN_LAYER_SIZE];
	float z_h[HIDDEN_LAYER_SIZE];

	float w_o[HIDDEN_LAYER_SIZE][v.dim];
	float b_o[v.dim];
	float z_o[v.dim];

	float d_h[HIDDEN_LAYER_SIZE];
	float d_o[v.dim];

	// random initialization
	for (int i = 0; i < v.dim; i++) {
		for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
			w_h[i][j] = random_float();
		}
	}

	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		b_h[i] = random_float();
	}

	for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
		for (int j = 0; j < v.dim; j++) {
			w_o[i][j] = random_float();
		}
	}

	for (int i = 0; i < v.dim; i++) {
		b_o[i] = random_float();
	}
	
	// training
	// train: data_train[i][0], target data_train[i][1]
	for (int e = 0; e < EPOCHS; e++) {
		for (int s = 0; s < sample_size; s++) {
			// feed forward
			// first layer
			for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
				z_h[i] = b_h[i];
				for (int j = 0; j < v.dim; j++) {
					z_h[i] += w_h[i][j] * (encode(&v, train_data[s][0])->vect[j]);
 				}
			}

			// output layer
			for (int i = 0; i < v.dim; i++) {
				z_o[i] = b_o[i];
				for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
					z_o[i] += w_o[i][j] * z_h[j];
				}
			}
			
			// activation
			softmax(z_o, z_o, v.dim);

			// backprop
			// output layer
			for (int i = 0; i < v.dim; i++) {
				d_o[i] = (z_o[i] - (encode(&v, train_data[s][0])->vect[i])) * z_o[i] * (1 - z_o[i]);
			}

			for (int i = 0; i < v.dim; i++) {
				for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
					w_o[i][j] -= LEARNING_RATE * d_o[i] * z_h[j];
				}
			}
			
			for (int i = 0; i < v.dim; i++) {
				b_o[i] -= LEARNING_RATE * d_o[i];
			}

			// hidden layer
			int i, j;
			for (i = 0; i < HIDDEN_LAYER_SIZE; i++) {
                d_h[i] = 0.0;
                for (j = 0; j < v.dim; j++) {
                    d_h[i] += d_o[j] * w_o[i][j];
                }
                d_h[i] *= z_h[i] * (1 - z_h[i]);
            }

            for (i = 0; i < HIDDEN_LAYER_SIZE; i++) {
                for (j = 0; j < v.dim; j++) {
                    w_h[i][j] -= LEARNING_RATE * d_h[i] * (encode(&v, train_data[s][0])->vect[j]);
                }
            }

            for (i = 0; i < HIDDEN_LAYER_SIZE; i++) {
                b_h[i] -= LEARNING_RATE * d_h[i];
            }
		}
	}

	// the weights of the hidden layer are the embeddings for the words

}

void createVocabulary(char file_path[], Vocabulary *v) {
	FILE *fp = fopen(file_path, "r");
	fscanf(fp, "%d\n", &(v->dim));
	char buff[MAX_WORD_SIZE];

	for (int i = 0; i < v->dim; i++) {
		fscanf(fp, "%s\n", buff);
		strcpy(v->voc[i].word, buff);
	}

	// one hot encoding
	for (int i = 0; i < v->dim; i++) {
		for (int j = 0; j < v->dim; j++) {
			if (i == j)
				v->voc[i].vect[j] = 1.0;
			else
				v->voc[i].vect[j] = 0.0;
		}
	}
}

int tokenize(char tokens[][MAX_WORD_SIZE], char str[]) {
	// tokenize a sentence and return the number of token
	int count = 0;	
	char *buff;

	buff = strtok(str, " ");
	while (buff != NULL) {
		strcpy(tokens[count], buff);
		count++;
		buff = strtok(NULL, " ");
	}

	return count;
}

VocItem *encode(Vocabulary *v, char word[]) {
	for (int i = 0; i < v->dim; i++) {
		if (strcmp(word, v->voc[i].word) == 0) {
			return &(v->voc[i]);
		}
	}
}

float softmax(float input[], float output[], int dim) {
	float m = -INFINITY;
	for (int i = 0; i < dim; i++) {
		if (input[i] > m)
			m = input[i];
	}

	float sum = 0.0;
	for (int i = 0; i < dim; i++) {
		sum += expf(input[i] - m);
	}

	float offset = m + logf(sum);
	for (int i = 0; i < dim; i++) {
		output[i] = expf(input[i] - offset);
	}
}

float random_float() {
	// generate a random float in the range [-1, 1]
    return -1 + 2.0 * ((float)rand() / (float)RAND_MAX);
}

float distance(float vectA[], float vectB[], int dim) {
	// calculate the distance between two vectors given the dimention
	float sum = 0.0;
	float d;
	
	for (int i = 0; i < dim; i++) {
		d = vectA[i] - vectB[i];
		sum += d * d;
	}

	return sqrt(sum);
}
