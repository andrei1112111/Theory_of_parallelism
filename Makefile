CXX = clang++
CXXFLAGS = -std=c++17 -Wall -O2
BUILD_DIR = build
OBJ = $(BUILD_DIR)/obj/main.o

TARGET = my_program
SRC = main.cpp

# ~~~   TO USE DOUBLE   ~~~
CALC_MODE = USE_DOUBLE

# ~~~   TO USE FLOAT    ~~~
#CALC_MODE =


$(shell mkdir -p $(BUILD_DIR))
$(shell mkdir -p $(BUILD_DIR)/obj)

ifeq ($(CALC_MODE),USE_DOUBLE)
CXXFLAGS += -D$(CALC_MODE)
endif

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(BUILD_DIR)/$(TARGET) $(OBJ)

$(BUILD_DIR)/obj/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(BUILD_DIR)
