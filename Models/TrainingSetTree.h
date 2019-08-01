#ifndef TREE_HEADER 
#define TREE_HEADER 
#include <vector>

template <typename T>
inline int decisionTree(const std::vector<T>& featureVector) { 
    if (featureVector[2] < 21842500.0) {
        if (featureVector[3] < 28.5) {
            if (featureVector[1] < 175.0) {
                if (featureVector[0] < 3.0) {
                    return 2.0;
                } 
                else {
                    return 1.0;
                } 
            } 
            else {
                if (featureVector[0] < 3.0) {
                    if (featureVector[2] < 56250.0) {
                        return 6.0;
                    } 
                    else {
                        if (featureVector[1] < 87500.0) {
                            if (featureVector[1] < 37625.0) {
                                return 8.0;
                            } 
                            else {
                                return 9.0;
                            } 
                        } 
                        else {
                            return 10.0;
                        } 
                    } 
                } 
                else {
                    if (featureVector[1] < 87500.0) {
                        if (featureVector[0] < 15.0) {
                            if (featureVector[0] < 5.0) {
                                if (featureVector[1] < 12625.0) {
                                    return 2.0;
                                } 
                                else {
                                    return 3.0;
                                } 
                            } 
                            else {
                                if (featureVector[0] < 7.0) {
                                    return 2.0;
                                } 
                                else {
                                    if (featureVector[0] < 9.0) {
                                        if (featureVector[2] < 99875.0) {
                                            return 1.0;
                                        } 
                                        else {
                                            return 6.0;
                                        } 
                                    } 
                                    else {
                                        return 2.0;
                                    } 
                                } 
                            } 
                        } 
                        else {
                            if (featureVector[1] < 62500.0) {
                                return 1.0;
                            } 
                            else {
                                return 2.0;
                            } 
                        } 
                    } 
                    else {
                        if (featureVector[0] < 13.0) {
                            return 3.0;
                        } 
                        else {
                            return 2.0;
                        } 
                    } 
                } 
            } 
        } 
        else {
            if (featureVector[3] < 93.5) {
                if (featureVector[1] < 375.0) {
                    if (featureVector[0] < 11.0) {
                        if (featureVector[0] < 9.0) {
                            if (featureVector[0] < 3.0) {
                                return 10.0;
                            } 
                            else {
                                if (featureVector[0] < 5.0) {
                                    return 8.0;
                                } 
                                else {
                                    if (featureVector[0] < 7.0) {
                                        return 7.0;
                                    } 
                                    else {
                                        return 8.0;
                                    } 
                                } 
                            } 
                        } 
                        else {
                            return 7.0;
                        } 
                    } 
                    else {
                        if (featureVector[0] < 13.0) {
                            return 6.0;
                        } 
                        else {
                            return 5.0;
                        } 
                    } 
                } 
                else {
                    if (featureVector[0] < 7.0) {
                        if (featureVector[0] < 5.0) {
                            if (featureVector[3] < 54.5) {
                                if (featureVector[0] < 3.0) {
                                    return 8.0;
                                } 
                                else {
                                    return 6.0;
                                } 
                            } 
                            else {
                                if (featureVector[0] < 3.0) {
                                    return 10.0;
                                } 
                                else {
                                    return 8.0;
                                } 
                            } 
                        } 
                        else {
                            return 6.0;
                        } 
                    } 
                    else {
                        if (featureVector[0] < 11.0) {
                            return 4.0;
                        } 
                        else {
                            if (featureVector[3] < 54.5) {
                                if (featureVector[0] < 15.0) {
                                    if (featureVector[1] < 625.0) {
                                        return 6.0;
                                    } 
                                    else {
                                        return 4.0;
                                    } 
                                } 
                                else {
                                    return 3.0;
                                } 
                            } 
                            else {
                                return 3.0;
                            } 
                        } 
                    } 
                } 
            } 
            else {
                if (featureVector[0] < 15.0) {
                    if (featureVector[2] < 875000.0) {
                        if (featureVector[0] < 13.0) {
                            if (featureVector[3] < 164.0) {
                                if (featureVector[0] < 9.0) {
                                    if (featureVector[0] < 7.0) {
                                        if (featureVector[2] < 531250.0) {
                                            return 10.0;
                                        } 
                                        else {
                                            return 8.0;
                                        } 
                                    } 
                                    else {
                                        return 9.0;
                                    } 
                                } 
                                else {
                                    if (featureVector[1] < 625.0) {
                                        if (featureVector[0] < 11.0) {
                                            return 7.0;
                                        } 
                                        else {
                                            return 6.0;
                                        } 
                                    } 
                                    else {
                                        return 7.0;
                                    } 
                                } 
                            } 
                            else {
                                if (featureVector[0] < 11.0) {
                                    if (featureVector[0] < 9.0) {
                                        if (featureVector[0] < 7.0) {
                                            return 10.0;
                                        } 
                                        else {
                                            return 9.0;
                                        } 
                                    } 
                                    else {
                                        if (featureVector[1] < 375375.0) {
                                            return 7.0;
                                        } 
                                        else {
                                            return 6.0;
                                        } 
                                    } 
                                } 
                                else {
                                    return 9.0;
                                } 
                            } 
                        } 
                        else {
                            if (featureVector[1] < 625.0) {
                                return 10.0;
                            } 
                            else {
                                if (featureVector[3] < 164.0) {
                                    return 6.0;
                                } 
                                else {
                                    return 8.0;
                                } 
                            } 
                        } 
                    } 
                    else {
                        if (featureVector[0] < 13.0) {
                            if (featureVector[0] < 11.0) {
                                if (featureVector[0] < 9.0) {
                                    return 9.0;
                                } 
                                else {
                                    if (featureVector[2] < 3625000.0) {
                                        return 10.0;
                                    } 
                                    else {
                                        return 5.0;
                                    } 
                                } 
                            } 
                            else {
                                if (featureVector[2] < 3625000.0) {
                                    return 8.0;
                                } 
                                else {
                                    return 10.0;
                                } 
                            } 
                        } 
                        else {
                            if (featureVector[2] < 6748750.0) {
                                return 10.0;
                            } 
                            else {
                                return 4.0;
                            } 
                        } 
                    } 
                } 
                else {
                    if (featureVector[3] < 216.5) {
                        if (featureVector[3] < 170.5) {
                            if (featureVector[2] < 531250.0) {
                                return 8.0;
                            } 
                            else {
                                if (featureVector[1] < 1625.0) {
                                    return 5.0;
                                } 
                                else {
                                    return 10.0;
                                } 
                            } 
                        } 
                        else {
                            return 7.0;
                        } 
                    } 
                    else {
                        return 9.0;
                    } 
                } 
            } 
        } 
    } 
    else {
        return 1.0;
    } 
}
#endif