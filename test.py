import numpy as np


img = np.array([[0,1,0,0,1],[0,1,1,0,0],[0,0,1,0,1],[0,0,0,0,0],[0,1,0,0,0]])

shape = np.shape(img)
print('-'*20)
print('Image:')
print(img)
print('-'*20)

row_first_bool = True
row_first=0
row_last=0
for i in range(shape[0]):
    if any(img[i,:]):
        if row_first_bool:
            row_first = i
            row_first_bool = False
        row_last = i


print(f'First Row: {row_first}')
print(f'Last Row: {row_last}')
print('-'*20)
col_first_bool = True
col_first=0
col_last=0
for i in range(shape[0]):
    if any(img[:,i]):
        if col_first_bool:
            col_first = i
            col_first_bool = False
        col_last = i
print(f'First Col: {col_first}')
print(f'Last Col: {col_last}')

print('-'*20)
crop = img[row_first:row_last+1,col_first:col_last+1]
print(crop)
print(np.shape(crop))
print('-'*20)
