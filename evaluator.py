
# calculate the misclassification ff
def evaluate_ff(P_class_A, P_notclass_A):
    P1 = P_class_A - P_notclass_A

    return P1
