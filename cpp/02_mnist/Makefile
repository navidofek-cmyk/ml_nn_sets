CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -I..
TARGET   = nn_mnist

$(TARGET): main.cpp data_loader.hpp ../common/matrix.hpp \
           ../common/activations.hpp ../common/losses.hpp ../common/network.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) main.cpp -lm

clean:
	rm -f $(TARGET)

.PHONY: clean
