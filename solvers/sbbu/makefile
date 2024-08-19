# Compiler
CXX := g++

# Source files
SRCS := sbbu.cpp
HEADERS := sbbu.h nmr.h vec3.h ddgp.h edge.h

# Compiler flags
CXXFLAGS_DEBUG := -Wall -Wextra -g -O0 -DDEBUG
CXXFLAGS_RELEASE := -Wall -Wextra -O3

# Libraries
LIBS := -lboost_regex -lm -fopenmp

# Targets
all: sbbu.exe sbbu_dbg.exe

sbbu.exe: $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS_RELEASE) -o $@ $(SRCS) $(LIBS)

sbbu_dbg.exe: $(SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS_DEBUG) -o $@ $(SRCS) $(LIBS)

test_all: sbbu.exe
	python call_sbbu.py

debug: sbbu_dbg.exe
	gdb ./sbbu_dbg.exe

.PHONY: all test_all debug clean

clean:
	rm -f sbbu.exe sbbu_dbg.exe