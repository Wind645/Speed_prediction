# d=float(input())
d=0.990
n=5
# n=int(input())
# m1,m2=map(float,input().split())
#m1=213.13
#m1=240.56
#m1=228.07
m1=240.52
m2=104.19
for i in range(n):
    t1,t2,t3=map(float,input().split())
    v1=d/t1
    v2=d/t2
    v3=d/t3
    print(v1,v2,v3)
    print(m1*v1,m2*v2+m1*v3)