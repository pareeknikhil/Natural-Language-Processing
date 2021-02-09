import operator

t1 = (1,2,3)
t2 = (2,3,4)
t3 = tuple(map(operator.add, t1, t2))
t4 = tuple(map(lambda x: x*-1, t3))
print(t3,t4)
te1 = ["1","2","3"]
test1 = tuple(float(i) for i in te1)
print(test1)
print(t1[0])