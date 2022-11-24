#encoding=UTF-8
import csv
with open(r'.\Iris数据集\iris.csv') as f:
    f_csv=csv.reader(f)
    headers = next(f_csv)
    print(headers[1:6])
    x=[]
    label=[]
    for row in f_csv:
        # print(row)
        x.append(row[1:5])
        if row[5]=='setosa':
            label.append(0)
        elif row[5]=='versicolor':
            label.append(1)
        else:
            label.append(2)
    print(x)
    print(label)

