#ifndef TREE_HEADER 
#define TREE_HEADER 
#include <vector>

template <typename T>
inline int decisionTree(const std::vector<T>& featureVector) { 
    if (featureVector[2] < 21842500.0) {
        if (featureVector[3] < 28.5) {
            if (featureVector[1] < 175.0) {
                return 1.0;
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
                    if (featureVector[2] < 112375.0) {
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
                                    if (featureVector[2] < 56250.0) {
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
                            if (featureVector[1] < 12625.0) {
                                if (featureVector[0] < 15.0) {
                                    return 2.0;
                                } 
                                else {
                                    return 1.0;
                                } 
                            } 
                            else {
                                if (featureVector[1] < 37500.0) {
                                    return 1.0;
                                } 
                                else {
                                    if (featureVector[0] < 9.0) {
                                        return 2.0;
                                    } 
                                    else {
                                        if (featureVector[1] < 62500.0) {
                                            return 4.0;
                                        } 
                                        else {
                                            return 2.0;
                                        } 
                                    } 
                                } 
                            } 
                        } 
                    } 
                    else {
                        if (featureVector[0] < 11.0) {
                            return 6.0;
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
                if (featureVector[0] < 5.0) {
                    if (featureVector[3] < 150.5) {
                        if (featureVector[1] < 375.0) {
                            return 9.0;
                        } 
                        else {
                            if (featureVector[3] < 134.5) {
                                if (featureVector[3] < 39.5) {
                                    if (featureVector[0] < 3.0) {
                                        return 8.0;
                                    } 
                                    else {
                                        return 4.0;
                                    } 
                                } 
                                else {
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
                            } 
                            else {
                                return 9.0;
                            } 
                        } 
                    } 
                    else {
                        return 10.0;
                    } 
                } 
                else {
                    if (featureVector[3] < 62.5) {
                        if (featureVector[0] < 11.0) {
                            if (featureVector[2] < 811875.0) {
                                if (featureVector[0] < 7.0) {
                                    if (featureVector[1] < 125250.0) {
                                        return 3.0;
                                    } 
                                    else {
                                        return 5.0;
                                    } 
                                } 
                                else {
                                    return 4.0;
                                } 
                            } 
                            else {
                                if (featureVector[0] < 7.0) {
                                    return 4.0;
                                } 
                                else {
                                    return 3.0;
                                } 
                            } 
                        } 
                        else {
                            return 3.0;
                        } 
                    } 
                    else {
                        if (featureVector[3] < 1737.5) {
                            if (featureVector[3] < 216.5) {
                                if (featureVector[3] < 170.5) {
                                    if (featureVector[2] < 1280750.0) {
                                        if (featureVector[0] < 11.0) {
                                            return 8.0;
                                        } 
                                        else {
                                            return 6.0;
                                        } 
                                    } 
                                    else {
                                        if (featureVector[1] < 1750.0) {
                                            return 4.0;
                                        } 
                                        else {
                                            return 5.0;
                                        } 
                                    } 
                                } 
                                else {
                                    if (featureVector[0] < 11.0) {
                                        if (featureVector[0] < 9.0) {
                                            return 8.0;
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
                            else {
                                if (featureVector[0] < 11.0) {
                                    if (featureVector[2] < 3625000.0) {
                                        if (featureVector[0] < 9.0) {
                                            return 9.0;
                                        } 
                                        else {
                                            return 10.0;
                                        } 
                                    } 
                                    else {
                                        return 6.0;
                                    } 
                                } 
                                else {
                                    if (featureVector[2] < 3625000.0) {
                                        return 8.0;
                                    } 
                                    else {
                                        return 6.0;
                                    } 
                                } 
                            } 
                        } 
                        else {
                            if (featureVector[0] < 11.0) {
                                return 5.0;
                            } 
                            else {
                                return 10.0;
                            } 
                        } 
                    } 
                } 
            } 
            else {
                if (featureVector[3] < 216.5) {
                    if (featureVector[3] < 170.5) {
                        if (featureVector[2] < 843375.0) {
                            if (featureVector[3] < 47.0) {
                                return 3.0;
                            } 
                            else {
                                if (featureVector[1] < 375.0) {
                                    if (featureVector[0] < 15.0) {
                                        return 6.0;
                                    } 
                                    else {
                                        return 5.0;
                                    } 
                                } 
                                else {
                                    if (featureVector[0] < 15.0) {
                                        if (featureVector[2] < 531250.0) {
                                            return 5.0;
                                        } 
                                        else {
                                            return 6.0;
                                        } 
                                    } 
                                    else {
                                        if (featureVector[3] < 124.0) {
                                            return 4.0;
                                        } 
                                        else {
                                            return 9.0;
                                        } 
                                    } 
                                } 
                            } 
                        } 
                        else {
                            if (featureVector[1] < 1750.0) {
                                if (featureVector[1] < 875.0) {
                                    if (featureVector[0] < 15.0) {
                                        return 4.0;
                                    } 
                                    else {
                                        return 3.0;
                                    } 
                                } 
                                else {
                                    return 3.0;
                                } 
                            } 
                            else {
                                if (featureVector[0] < 15.0) {
                                    return 4.0;
                                } 
                                else {
                                    return 10.0;
                                } 
                            } 
                        } 
                    } 
                    else {
                        if (featureVector[0] < 15.0) {
                            return 8.0;
                        } 
                        else {
                            return 7.0;
                        } 
                    } 
                } 
                else {
                    if (featureVector[0] < 15.0) {
                        return 10.0;
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