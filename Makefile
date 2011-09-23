CXX := g++ -w
INCLUDES := -I. -I/usr/include/ni -I/usr/local/include

all:  demo record convert.o playback.o visualize.o

demo: demo.cpp convert.o playback.o visualize.o filter.o
	$(CXX) demo.cpp convert.o playback.o visualize.o filter.o $(INCLUDES) -o demo -L/usr/local/lib/ -lopencv_highgui -lopencv_imgproc -lopencv_core -L/usr/lib/ -lOpenNI

record: record.cpp convert.o playback.o
	$(CXX)  record.cpp convert.o playback.o $(INCLUDES) -o record -L/usr/local/lib/ -lopencv_highgui -lopencv_imgproc -lopencv_core -L/usr/lib/ -lOpenNI

convert.o: convert.cpp
	$(CXX) -c convert.cpp $(INCLUDES) -o convert.o

playback.o: playback.cpp
	$(CXX) -c playback.cpp $(INCLUDES) -o playback.o

visualize.o: visualize.cpp
	$(CXX) -c visualize.cpp $(INCLUDES) -o visualize.o

filter.o: filter.cpp
	$(CXX) -c filter.cpp $(INCLUDES) -o filter.o
clean:
	\rm -f record demo convert.o playback.o visualize.o filter.o
