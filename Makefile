
ROOTCFLAGS   := $(shell root-config --cflags)
ROOTLIBS     := $(shell root-config --libs)
ROOTINCLUDE  := -I$(shell root-config --incdir)

all: deep-pipi-gen

deep-pipi-gen:
	$(CXX) -O3 $(ROOTINCLUDE) $(ROOTCFLAGS) -o deep-pipi-gen Deep_pipi.cpp $(ROOTLIBS)

clean:
	rm -rf deep-pipi-gen
