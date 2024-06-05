#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "util.h"

// TODO: create functions to encode / decode words to embeddings

#define WINDOW 4
#define HIDDEN_LAYER_SIZE 5
#define EPOCHS 100
#define LEARNING_RATE 0.1

int main() {
	Vocabulary v;
	Sentence data[MAX_SENTENCE_NUMBER];

	// create the vocabulary
	createVocabulary(VOC_FILE_PATH, &v);

	// tokenize the data
	create_data(data);

	// create training samples
	int sample_size = 0;
	char train_data[50][2][MAX_WORD_SIZE];

	FILE *fp = fopen(DATA_FILE_PATH, "r");
	int sentence_count;
	fscanf(fp, "%d\n", &sentence_count);

	for (int i = 0; i < sentence_count; i++) {
		for (int j = 0; j < data[i].dim; j++) {
			for (int k = j+1; k < data[i].dim && k < j + 1 + WINDOW; k++) {
				// create couple of words
				strcpy(train_data[sample_size][0], data[i].words[j]);
				strcpy(train_data[sample_size][1], data[i].words[k]);
				sample_size++;
			}
		}
	}

	// create the nn for embeddings
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
	// train: data_train[i][0], target: data_train[i][1]
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
			// calculate deltas
			for (int i = 0; i < v.dim; i++) {
				d_o[i] = (z_o[i] - (encode(&v, train_data[s][0])->vect[i])) * z_o[i] * (1 - z_o[i]);
			}
			// adjust weights
			for (int i = 0; i < v.dim; i++) {
				for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
					w_o[i][j] -= LEARNING_RATE * d_o[i] * z_h[j];
				}
			}
			// adjust biases
			for (int i = 0; i < v.dim; i++) {
				b_o[i] -= LEARNING_RATE * d_o[i];
			}

			// hidden layer
			// calculate deltas
			for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
                d_h[i] = 0.0;
                for (int j = 0; j < v.dim; j++) {
                    d_h[i] += d_o[j] * w_o[i][j];
                }
                d_h[i] *= z_h[i] * (1 - z_h[i]);
            }
			// adjust weight
            for (i = 0; i < HIDDEN_LAYER_SIZE; i++) {
                for (j = 0; j < v.dim; j++) {
                    w_h[i][j] -= LEARNING_RATE * d_h[i] * (encode(&v, train_data[s][0])->vect[j]);
                }
            }
			// asdjust biases
            for (i = 0; i < HIDDEN_LAYER_SIZE; i++) {
                b_h[i] -= LEARNING_RATE * d_h[i];
            }
		}
	}

	// the weights of the hidden layer are the embeddings for the words
	// example
	printf("rome-italy: %f\n", distance(w_h[13], w_h[14], HIDDEN_LAYER_SIZE));
	printf("rome-france: %f\n", distance(w_h[13], w_h[12], HIDDEN_LAYER_SIZE));

}
