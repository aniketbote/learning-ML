#p = [0,1,0,0,0]
p = [0.2,0.2,0.2,0.2,0.2]
world = ['green','red','red','green','green']
measurements = ['red','red']
motions = [1,1]
phit = 0.6
pmiss = 0.2
pExact = 0.8
pOvershoot = 0.1
pUndershoot = 0.1
def sense(p,z):
    q = []
    for i in range(len(p)):
        hit = (z == world[i])
        q.append(p[i] * (hit * phit + (1-hit)*pmiss))
        s = sum(q)
    q = list(map(lambda x : x / s, q))
    return q
for i in range(len(measurements)):
    p = sense(p,measurements[i])
def move(p,U):
    q = []
    for i in range(len(p)):
        s = p[(i-U)%len(p)]*pExact
        s = s + pOvershoot * p[(i-U-1)%len(p)]
        s = s + pUndershoot * p[(i-U+1)%len(p)]
        q.append(s)
    return q
for i in range(len(measurements)):
    p = sense(p,measurements[i])
    p = move(p,motions[i])
print(p)
