import os

arr = os.listdir('test_images')

with open('csvfile.csv','wb') as file:
    arr.sort()
    for a in arr: 
        print(a)   
        file.write(a)
        file.write('\n')