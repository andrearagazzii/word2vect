#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"

typedef struct {
	int dim;
	char words[MAX_WORD_NUMBER][MAX_WORD_SIZE];
} Sentence;

void remove_newline_ch(char *line) {
	int new_line = strlen(line) - 1;
	if (line[new_line] == '\n')
		line[new_line] = '\0';
}

// TODO: from a file with some sentences (first row number of sentences)
// 1. create a file with unique words (voc.txt)
// 2. function to tokenize each sentence

int main(int argc, char *argv[]) {
	FILE *fp = fopen("data.txt", "r");

	Sentence data[MAX_SENTENCE_NUMBER];

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

	char unique_word[MAX_WORD_NUMBER][MAX_WORD_SIZE];
	int unique_word_count = 0;

	for (int i = 0; i < sentence_count; i++) {
		for (int j = 0; j < data[i].dim; j++) {
			int add = 1;
			for (int k = 0; k < unique_word_count; k++) {
				if (strcmp(data[i].words[j], unique_word[k]) == 0) {
					add = 0;
				}
			}
			if (add) {
				strcpy(unique_word[unique_word_count], data[i].words[j]);
				unique_word_count++;
			}
		}
	}

	fprintf(fp, "%d\n", unique_word_count);

	for (int i = 0; i < unique_word_count; i++) {
		fprintf(fp, "%s\n", unique_word[i]);
	}


	return 0;
}
