# Makefile for compiling the BP program

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -Wextra -std=c11 -O2

# Debug flags
DEBUG_FLAGS = -g -O0

# Libraries
LIBS = -lblas -llapack -llapacke -lm

# Target executable
TARGET = bp.exe

# Source files
SRCS = bp.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Rule for creating the target executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LIBS)

# Rule for compiling C source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Debug target
debug: CFLAGS := $(CFLAGS) $(DEBUG_FLAGS)
debug: clean $(TARGET)

run: all
	./$(TARGET) dmdgp/1a11_model1_chainA_segment0.csv -v

# Rule for cleaning up generated files
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean debug
