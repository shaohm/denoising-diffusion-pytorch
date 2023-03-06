import math
def pos_vector(k, d):
    assert(d % 2 == 0)
    v = []
    for i in range(d//2):
        denom = 10000 ** (2*i/d)
        v.append(math.sin(k/denom))
        v.append(math.cos(k/denom))

    v = ["%+.4f" % a for a in v]
    return v

d = 10
for k in range(20):
    print(pos_vector(k,d))
