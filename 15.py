#三数之和：输入一组数，求其中三个数字之和为0的所有组合，不包括重复的且每个数只能用一次
x=input("please input numbers:")
x=x.split(',')
for i in range(len(x)):
    x[i]=x[i].strip()
result=[]
for i in range(len(x)):
    for j in range(i+1,len(x)):
        y=int(x[i])+int(x[j])
        p=[int(x[i]),int(x[j]),-y]
        p.sort()
        if ((str(-y)) in x[j+1:len(x)]) and (not(p in result)):
            result.append(p)
print(result)