import numpy as np
import random
import copy

'''
Search space: FOX-NAS-CPU
Imput image size: 192
'''

# Accuracy predictor and latency predictor
acc0 = 61.878882
latency0 = 16.30464
acc_array = np.array([0.308031, 0.246401, 0.385460, 0.549228, 0.355705, 0.020765, 0.041229, 0.034954, 0.067013, 0.135598, 0.066404, -0.012060, 0.054933, 0.009343, 0.014635, 0.057120, 0.043826, 0.056370, 0.089531, 0.113646, 0.005593, 0.025202, 0.027489, 0.003217, 0.013580, 0.088727, 0.116691, 0.120979, 0.203725, 0.153543, -0.020289, -0.026291, -0.027556, -0.040638, -0.031194, -0.014607, 0.006102, -0.006670, 0.004494, 0.002630])
latency_array = np.array([-8.06211,-2.69638,-1.13344  ,-0.22013,-0.55329,-2.32253,-0.90273  ,-1.01517,-0.54179,-0.30544,-2.33407  ,-0.94015,-0.11907,0.01387,-0.27301  ,2.85597,1.29418,0.65506,0.68359  ,0.62364,1.17867,0.49333,0.10513  ,0.23057,0.01159,2.15201,1.01542  ,0.70170,0.87146,0.45674,1.91980  ,0.70742,0.16173,0.21330,0.17648])

# Architecture candidate
tot_candidate = [2,3,4]
e_candidate = [2,3,4,6]
k_candidate = [3,5,7]

# Architecture weight
arr_weight = np.array([100,100,100,100,100,
                       1,0,0,1,1,
                       0,0,0,0,0,
                       0,0,0,0,1,
                       0,0,0,0,0])
arr_weight = arr_weight / sum(arr_weight)


# Latency constraint
target_latency = 55

def get_latency(x):
    return latency0 + sum(x * latency_array)

def get_acc(x):
    return acc0 + sum(x * acc_array)

def init():
    arr = []
    arr_idx = []
    
    for tot in range(5):
        tmp_idx = np.random.randint(3)
        arr.append(tot_candidate[tmp_idx])
        arr_idx.append(tmp_idx)
    for avg_e in range(5):
        #tmp_idx = np.random.randint(4)
        tmp_idx = 0
        arr.append(e_candidate[tmp_idx])
        arr_idx.append(tmp_idx)
    for avg_k_24 in range(5):
        #tmp_idx = np.random.randint(3)
        tmp_idx = 0
        arr.append(k_candidate[tmp_idx])
        arr_idx.append(tmp_idx)
    for tot_e_16_24 in range(5):
        #tmp_idx = np.random.randint(4)
        tmp_idx = 0
        arr.append(e_candidate[tmp_idx])
        arr_idx.append(tmp_idx)
    for tot_k_16_24 in range(5):
        #tmp_idx = np.random.randint(3)
        tmp_idx = 0
        arr.append(k_candidate[tmp_idx])
        arr_idx.append(tmp_idx)
        
    return arr, arr_idx

def get_arr(tmp_x):
    x_latency = copy.deepcopy(tmp_x)
    x_acc = copy.deepcopy(tmp_x)
    for i in range(5):
        x_latency.append(tmp_x[i] * tmp_x[i+5])  # tot_24 * avg_e_24
        x_acc.append((tmp_x[i] - 1) * tmp_x[i+5])  # (tot_24 - 1) * avg_e_24
    for i in range(5):
        x_latency.append(tmp_x[i] * tmp_x[i+10])  # tot_24 * avg_k_24
        x_acc.append(tmp_x[i] * (tmp_x[i] - 1) * tmp_x[i+5])  # tot_24 * (tot_24 - 1) * avg_e_24

    x_latency = np.array(x_latency)
    x_acc = np.array(x_acc)
    x_acc = np.hstack((x_acc, x_latency[-5:]))  # (tot_24 - 1) * avg_k_24
    
    return x_acc, x_latency


record_x = []
npz_t = []
npz_acc = []
def main(print_bool = False):
    global record_x, target_latency, arr_weight
    
    latency = 999
    x_latency = []
    while latency > target_latency:
        tmp_x, tmp_x_idx = init()
        x_acc, x_latency = get_arr(tmp_x)
        latency = get_latency(x_latency)
        acc = get_acc(x_acc)

    len_x = [i for i in range(len(tmp_x_idx))]
    step = 8
    t = 0
    t2 = 0
    acc_arr = []
    latency_arr = []
    tmp_x_arr = []
    x_latency_arr = []
    tmp_npz_t = []
    tmp_npz_acc = []
    move_candidate = (1,0,-1)
    while step > 0:
        next_latency = 999
        while next_latency > 55:
            next_tmp_x = copy.deepcopy(tmp_x)
            next_tmp_x_idx = copy.deepcopy(tmp_x_idx)
            move_idx = np.random.choice(len_x, p=arr_weight, size=step, replace=False)
            for idx in move_idx:
                if idx < 5:
                    candidate = tot_candidate
                elif idx < 10 or ((idx > 14) and (idx < 20)):
                    candidate = e_candidate
                else:
                    candidate = k_candidate
                   
                move = move_candidate[np.random.randint(3)]
                while ( next_tmp_x_idx[idx] + move ) > ( len(candidate) - 1 ) or (( next_tmp_x_idx[idx] + move ) < 0):
                    move = move_candidate[np.random.randint(3)]
                next_tmp_x_idx[idx] = next_tmp_x_idx[idx] + move
                next_tmp_x[idx] = candidate[next_tmp_x_idx[idx]]
            next_x_acc, next_x_latency = get_arr(next_tmp_x)
            next_latency = get_latency(next_x_latency)
        next_acc = get_acc(next_x_acc)
        t += 1
        if ( next_acc > acc ) or (t2 > 500 and t < 10000 and ( next_acc - acc ) > -0.1):# or ( np.exp(( next_acc - acc ) * t) > threshold):
            tmp_x, tmp_x_idx, acc, latency, x_latency = next_tmp_x, next_tmp_x_idx, next_acc, next_latency, next_x_latency
            t2 = 0
            acc_arr.append(acc)
            latency_arr.append(latency)
            tmp_x_arr.append(tmp_x)
            record_x.append(x_latency)
            tmp_npz_t.append(t)
            tmp_npz_acc.append(acc)
        else:
            t2 += 1

        if print_bool:
            print('*' * 50)
            print(t)
            print(t2)
            print(acc)
            
        if t == 200:        
            move_candidate = (1,0,-1)
            arr_weight = np.array([1,1,1,1,1,
                                   1,3,3,3,1,
                                   1,3,3,3,3,
                                   1,3,3,3,1,
                                   1,3,3,3,3])
        arr_weight = arr_weight / sum(arr_weight)
        
        if step > 5:
            if t % 1000 == 0:
                step -= 1
                
        if t2 == 5000:
            break
    
    if print_bool:
        print(acc, latency)
        print(tmp_x)
    
    npz_t.append(tmp_npz_t)
    npz_acc.append(tmp_npz_acc)
    
    return acc_arr, latency_arr, t - t2
        
record_acc = np.array([])
record_latency = np.array([])
record_max_acc = np.array([])
record_t = np.array([])
for i in range(3):
    print(i,end='\r')
    results = main(True)
    record_acc = np.append(record_acc, results[0])
    record_latency = np.append(record_latency, results[1])
    record_max_acc = np.append(record_max_acc, results[0][-1])
    record_t = np.append(record_t, results[2])
record_x = np.array(record_x)

# Remove duplicate
record_acc_final = np.unique(record_acc)[-5:]
record_latency_final = []
record_x_final = []
for i in record_acc_final:
    record_latency_final.append(record_latency[ np.where( record_acc == i ) ][0])
    record_x_final.append(record_x[ np.where( record_acc == i ) ][0])



model_name = 'FOX-NAS-CPU'
img_size = '192'

np.save(model_name + '_' + img_size + '_best_arch', record_x_final)
np.save(model_name + '_' + img_size + '_best_acc', record_acc_final)
np.save(model_name + '_' + img_size + '_best_latency', record_latency_final) 

print('='*80 + '\nBest:')
print(record_acc_final[-1], record_latency_final[-1])
print(record_x_final[-1])




