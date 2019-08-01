#ifndef TREE_HEADER 
#define TREE_HEADER 
#include <vector>

inline int decisionTree(const std::vector<double>& featureVector) { 
    if (featureVector[1] < 375000.0) {
        if (featureVector[0] < 3.0) {
            if (featureVector[1] < 62500.0) {
                return 6.0;
            } 
            else {
                if (featureVector[1] < 87500.0) {
                    return 9.0;
                } 
                else {
                    return 10.0;
                } 
            } 
        } 
        else {
            if (featureVector[1] < 175000.0) {
                if (featureVector[0] < 7.0) {
                    if (featureVector[1] < 87500.0) {
                        if (featureVector[0] < 5.0) {
                            if (featureVector[1] < 62500.0) {
                                return 2.0;
                            } 
                            else {
                                return 3.0;
                            } 
                        } 
                        else {
                            if (featureVector[1] < 62500.0) {
                                return 3.0;
                            } 
                            else {
                                return 2.0;
                            } 
                        } 
                    } 
                    else {
                        if (featureVector[0] < 5.0) {
                            return 3.0;
                        } 
                        else {
                            return 2.0;
                        } 
                    } 
                } 
                else {
                    if (featureVector[0] < 9.0) {
                        return 2.0;
                    } 
                    else {
                        if (featureVector[1] < 62500.0) {
                            if (featureVector[0] < 13.0) {
                                return 2.0;
                            } 
                            else {
                                return 1.0;
                            } 
                        } 
                        else {
                            if (featureVector[0] < 13.0) {
                                if (featureVector[1] < 87500.0) {
                                    return 2.0;
                                } 
                                else {
                                    return 3.0;
                                } 
                            } 
                            else {
                                return 2.0;
                            } 
                        } 
                    } 
                } 
            } 
            else {
                if (featureVector[0] < 13.0) {
                    if (featureVector[0] < 11.0) {
                        if (featureVector[0] < 5.0) {
                            return 8.0;
                        } 
                        else {
                            if (featureVector[0] < 7.0) {
                                return 5.0;
                            } 
                            else {
                                return 4.0;
                            } 
                        } 
                    } 
                    else {
                        return 3.0;
                    } 
                } 
                else {
                    if (featureVector[0] < 15.0) {
                        return 5.0;
                    } 
                    else {
                        return 4.0;
                    } 
                } 
            } 
        } 
    } 
    else {
        if (featureVector[1] < 875000.0) {
            if (featureVector[0] < 13.0) {
                if (featureVector[0] < 7.0) {
                    return 10.0;
                } 
                else {
                    if (featureVector[1] < 625000.0) {
                        if (featureVector[0] < 11.0) {
                            if (featureVector[0] < 9.0) {
                                return 8.0;
                            } 
                            else {
                                return 7.0;
                            } 
                        } 
                        else {
                            return 6.0;
                        } 
                    } 
                    else {
                        if (featureVector[0] < 11.0) {
                            return 6.0;
                        } 
                        else {
                            return 9.0;
                        } 
                    } 
                } 
            } 
            else {
                if (featureVector[0] < 15.0) {
                    return 5.0;
                } 
                else {
                    if (featureVector[1] < 625000.0) {
                        return 4.0;
                    } 
                    else {
                        return 7.0;
                    } 
                } 
            } 
        } 
        else {
            if (featureVector[0] < 15.0) {
                if (featureVector[0] < 13.0) {
                    if (featureVector[0] < 11.0) {
                        return 10.0;
                    } 
                    else {
                        return 8.0;
                    } 
                } 
                else {
                    return 10.0;
                } 
            } 
            else {
                return 9.0;
            } 
        } 
    } 
}
#endif