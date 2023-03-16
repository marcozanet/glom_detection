import numpy as np


fp = '/Users/marco/prova.txt'
text = '8 9 10'
with open(fp, 'w') as f:
    f.writelines(text)

with open(fp, 'r') as f:
    text = f.readlines()

for row in text: 
    print(row)

print(type(text))
print(type(text[0]))
print(text)