#ifndef BASE_TYPES
#define BASE_TYPES


#include "magma_v2.h"


struct dim2 {
public:
    dim2() {}

    dim2(int first, int second) {
        values[0] = first;
        values[1] = second;
    }

    int& operator[](int position) {
        return values[position];
    }

private:
    int values[2] = {0, 0};
};


#endif
