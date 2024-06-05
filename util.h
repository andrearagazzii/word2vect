#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define MAX_WORD_SIZE 30
#define MAX_WORD_NUMBER 200
#define MAX_SENTENCE_NUMBER 10 
#define DATA_FILE_PATH "data.txt"
#define VOC_FILE_PATH "voc.txt"

typedef struct {
	char word[MAX_WORD_SIZE];
	float vect[MAX_WORD_NUMBER];
} VocItem;

typedef struct {
	int dim;
	VocItem voc[MAX_WORD_NUMBER];
} Vocabulary;

typedef struct {
	int dim;
	char words[MAX_WORD_NUMBER][MAX_WORD_SIZE];
} Sentence;

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

void remove_newline_ch(char *line) {
	int new_line = strlen(line) - 1;
	if (line[new_line] == '\n')
		line[new_line] = '\0';
}


void create_data(Sentence data[MAX_SENTENCE_NUMBER]) {
	FILE *fp = fopen("data.txt", "r");

	int sentence_count;
	fscanf(fp, "%d\n", &sentence_count);
	
	char curr_sent[MAX_WORD_NUMBER * MAX_WORD_SIZE];
	for (int i = 0; i < sentence_count; i++) {
		/* fscanf(fp, "%s\n", curr_sent); */
		fgets(curr_sent, sizeof(curr_sent), fp);
		data[i].dim = tokenize(data[i].words, curr_sent);
		remove_newline_ch(data[i].words[data[i].dim - 1]);
	}
	fclose(fp);
	
	// create the voc file
	fp = fopen("voc.txt", "w");

	char unique_words[MAX_WORD_NUMBER][MAX_WORD_SIZE];
	int unique_words_count = 0;

	for (int i = 0; i < sentence_count; i++) {
		for (int j = 0; j < data[i].dim; j++) {
			int add = 1;
			for (int k = 0; k < unique_words_count; k++) {
				if (strcmp(data[i].words[j], unique_words[k]) == 0) {
					add = 0;
				}
			}
			if (add) {
				strcpy(unique_words[unique_words_count], data[i].words[j]);
				unique_words_count++;
			}
		}
	}

	fprintf(fp, "%d\n", unique_words_count);

	for (int i = 0; i < unique_words_count; i++) {
		fprintf(fp, "%s\n", unique_words[i]);
	}
}
