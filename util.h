#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_WORD_SIZE 20
#define MAX_WORD_NUMBER 100
#define MAX_SENTENCE_NUMBER 10 

typedef struct {
	char word[MAX_WORD_SIZE];
	float vect[MAX_WORD_NUMBER];
} VocItem;

typedef struct {
	int dim;
	VocItem voc[MAX_WORD_NUMBER];
} Vocabulary;

void createVocabulary(char file_path[], Vocabulary *v);  
int tokenize(char tokens[][MAX_WORD_SIZE], char str[]); // split into words str
VocItem *encode(Vocabulary *v, char word[]);
void softmax(float input[], float output[], int dim);
float random_float(); // random float range [-1, 1]
float distance(float vectA[], float vectB[], int dim); // distance between two vectors

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
	return NULL;
}

void softmax(float input[], float output[], int dim) {
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
