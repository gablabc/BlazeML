#ifndef TREE_HEADER 
#define TREE_HEADER 
#include <vector>

inline int decisionTree(const std::vector<double>& featureVector) { 
    if (featureVector[1] < 0.4) {
        return 2;
    } 
    else {
        if (featureVector[0] < 1) {
            return 2        
        }
        else {
            return 1;
        } 
    }
}
#endif