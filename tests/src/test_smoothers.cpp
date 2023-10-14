#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <ValueSmoother.hpp>

using namespace std;

int main(void) {
    LinearValueSmoother param1Coeff;
    ExponentialValueSmoother param2Coeff;

    param1Coeff.setSampleRate(48000.0f);
    param1Coeff.setTimeConstant(0.1f);
    param1Coeff.setTargetValue(0.0f);
    param1Coeff.clearToTargetValue();

    param2Coeff.setSampleRate(48000.0f);
    param2Coeff.setTimeConstant(0.1f);
    param2Coeff.setTargetValue(0.0f);
    param2Coeff.clearToTargetValue();

    param1Coeff.setTargetValue(1.0f);
    param2Coeff.setTargetValue(1.0f);

    /* 1 sec */
    for(int i=0; i<48000u; i++) {
        param1Coeff.next();
        param2Coeff.next();
        if (i%100 == 0)
            printf("%d) %.02f %.02f\n", i, param1Coeff.getCurrentValue(), param2Coeff.getCurrentValue());
    }
}