CXX = g++
CXXFLAGS = -O3 -Wall -Wextra -I. -fopenmp -ffast-math -std=c++17

SRCS = $(wildcard src/*.cpp) $(wildcard src/*.c)

OBJS = $(SRCS:.cpp=.o)
TARGET = fluid

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.c
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET)
