#include <stdio.h>

int decision_tree(float X[]) {
    if (X[3] <= 0.800000) {
        return 0;
    } else {
        if (X[3] <= 1.750000) {
            if (X[2] <= 5.400000) {
                if (X[3] <= 1.450000) {
                    return 1;
                } else {
                    if (X[2] <= 4.950000) {
                        return 1;
                    } else {
                        if (X[1] <= 2.600000) {
                            return 2;
                        } else {
                            return 1;
                        }
                    }
                }
            } else {
                return 2;
            }
        } else {
            if (X[2] <= 4.850000) {
                if (X[1] <= 3.100000) {
                    return 2;
                } else {
                    return 1;
                }
            } else {
                return 2;
            }
        }
    }
}

int main() {
        float test_sample[] = {5.1f, 3.5f, 1.4f, 0.2f};  // Example iris sample
        int result = decision_tree(test_sample);
        printf("Predicted class: %d\n", result);
        return 0;
}
