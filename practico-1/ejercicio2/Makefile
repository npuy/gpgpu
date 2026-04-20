CC = gcc
CFLAGS = -O2 -Wall -std=c11
LDFLAGS = -lm

SRC_DIR = src
RESULTS_DIR = results

TARGET = ejercicio2

all: $(TARGET) | $(RESULTS_DIR)

$(TARGET): $(SRC_DIR)/ejercicio2.c $(SRC_DIR)/utils.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
	rm -f $(RESULTS_DIR)/*.csv

.PHONY: all run clean
