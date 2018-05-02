import os
import sys
import random
from shutil import copyfile
import shutil

path = sys.argv[1] + "/data"
count = 0
print(len([files for files in os.listdir(path)]))
if not os.path.exists(sys.argv[1] + "/newdata/"):
    os.mkdir (sys.argv[1]+"/newdata/")
for files in os.listdir(path):
    new_path = path + "/" + str(files)
    mid_files = [f for f in os.listdir(new_path)]
    length = len(mid_files)
    if length > 2:
        if not os.path.exists(sys.argv[1] + "/newdata/"+str(files)):
            os.mkdir (sys.argv[1]+"/newdata/"+str(files))
        test_files = random.sample(mid_files, int(length*0.1)+1)
        test_path = sys.argv[1] + "/newdata/" + str(files) + "/test"
        val_path = sys.argv[1] + "/newdata/" + str(files) + "/val"
        train_path = sys.argv[1] + "/newdata/" + str(files) + "/train"
        if not os.path.exists(sys.argv[1] + "/newdata/"+str(files)+"/test"):
            os.mkdir (sys.argv[1]+"/newdata/"+str(files)+"/test")
        for f in test_files:
            shutil.copy2(new_path+"/"+str(f), test_path)
        vals = [f for f in os.listdir(new_path) if f not in test_files]
        rand_vals = random.sample(vals, int(len(vals)*0.1)+1)
        if not os.path.exists(sys.argv[1] + "/newdata/"+str(files)+"/val"):
            os.mkdir (sys.argv[1]+"/newdata/"+str(files)+"/val")
        for f in rand_vals:
            shutil.copy2(new_path+"/"+str(f), val_path)
        train = [f for f in vals if f not in rand_vals]
        if not os.path.exists(sys.argv[1] + "/newdata/"+str(files)+"/train"):
            os.mkdir (sys.argv[1]+"/newdata/"+str(files)+"/train")
        for f in train:
            shutil.copy2(new_path+"/"+str(f), train_path)


    
