sbbu.exe: sbbu.cpp sbbu.h nmr.h vec3.h
	g++ -O3 -Wall sbbu.cpp -o sbbu.exe -fopenmp -lboost_regex -lm

test_all: sbbu.exe
	python call_sbbu.py