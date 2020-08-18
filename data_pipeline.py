import numpy as np

def hotkey(arr):
    temp = np.zeros((arr.size,arr.max()+1))
    temp[np.arange(arr.size),arr] = 1    
    return temp
    
def pipeline(batch_size,nbatchs,data,label):
    i = 0
    while (True):
        if(i >= nbatchs):
            print('Done')
            yield None,None
            break
        c_data  = data[i*batch_size:((i+1)*batch_size)].toarray()
        c_label = hotkey(label[i*batch_size:((i+1)*batch_size)])
        yield c_data,c_label
        i+1