CC = g++-14
CXXFLAGS = -lm -I. -I/usr/local/opt/libomp/include -Dcimg_display=0 -fopenmp

.PHONY: runfast
runfast: predict.c
	$(CC) -Ofast -o predict predict.c $(CXXFLAGS)

.PHONY: run
run: predict.c
	$(CC) -O3 -o predict predict.c $(CXXFLAGS)