CC = gcc
CFLAGS = -O2 -Wall -std=c11
LDFLAGS = -lm

SRC_DIR = src
RESULTS_DIR = results

TARGET = ejercicio1

all: $(TARGET) | $(RESULTS_DIR)

$(TARGET): $(SRC_DIR)/ejercicio1.c $(SRC_DIR)/utils.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

run: $(TARGET)
	./$(TARGET)

plots: run
	cd scripts && python3 generate_plots.py

clean:
	rm -f $(TARGET)
	rm -f $(RESULTS_DIR)/*.csv

.PHONY: all run plots clean
