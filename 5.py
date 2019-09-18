#a
#since false negative rate is zero, then there is no positive points splitting into negative. No wrong blues
def decision_tree_a(x1, x2):
    if x1 > 2:
        return 1
    else:
        return 0
def decision_tree_b(x1,x2):
    if x1 > 2:
        if x2 > 2:
            return 1
        else:
            return 0
    else:
        return 0