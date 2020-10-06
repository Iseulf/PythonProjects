#求数组中三数之和与目标数最接近的
def FindNumber(x,sum):
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            temp=sum-x[i]-x[j]
            if temp in x[j+1:len(x)]:
                return [x[i],x[j],temp]
    return []
x=input("please input numbers:")
goal=int(input("please input goal:"))
x=x.split(',')
y=[]
sign=1
for i in range(len(x)):
    y.append(int(x[i].strip()))
i=0
while sign==1:
    foobar=FindNumber(y,goal+i)
    if bool(foobar):
        sign=0
    else:
        foobar=FindNumber(y,goal-i)
        if bool(foobar):
            sign=0
    i=i+1
print(sum(foobar))