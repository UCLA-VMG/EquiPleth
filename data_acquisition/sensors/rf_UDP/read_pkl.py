import organizer_copy as org
import pickle
import sys
import numpy as np


file_root = sys.argv[1]
file_name = file_root+'.pkl'
f = open('./' + file_name,'rb')
print(f)
s = pickle.load(f)

o = org.Organizer(s, 1, 4, 3, 512)
frames = o.organize()

print(frames.shape)

to_save = {'frames':frames, 'start_time':s[3], 'end_time':s[4], 'num_frames':len(frames)}

with open('./' + file_root + '_read.pkl', 'wb') as f:
    pickle.dump(to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
