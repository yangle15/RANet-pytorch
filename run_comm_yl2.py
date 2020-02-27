import os
import plot_yl as p


def run_exp_cmd(scale, arch, compress_factor, stepmode, data, gpu, evalmode):
    if stepmode == 'even':
        s = 4
    elif stepmode =='lg':    
        if arch == 'ranet1':
            s = 1
        else:
            s = 2
    else:
        raise NotImplementedError

    if scale == '1-2-3-3':
        #gbf = '4-2-2-1'
        #modelname = f'{arch}-x4s{s}c{compress_factor}-{stepmode}'
        gbf = '4-2-1-1'
        modelname = f'{arch}-x3s{s}c{compress_factor}-{stepmode}'
    
    elif scale == '1-2-2-3':
        
        gbf = '4-2-2-1'
        scale = '1-2-2-3'
        modelname = f'{arch}-x4s{s}c{compress_factor}-{stepmode}'
        #modelname = f'{arch}-x5s{s}c{compress_factor}-{stepmode}'
    elif scale == '1-1-2-3':
        gbf = '4-4-2-1'
        modelname = f'{arch}-x2s{s}c{compress_factor}-{stepmode}'
    elif scale == '1-2-3':
        gbf = '4-2-1'
        modelname = f'{arch}-x1s{s}c{compress_factor}-{stepmode}'
    else:
        raise NotImplementedError
    print(modelname)

    
    save_dir = f'save/2020weight2/{data}-{modelname}'
    cmd = f'python main_v5_2.py --arch {arch}\
        --data-root /home/hanyz/msdnet-v5-no-use/data --data {data} --save {save_dir}\
        --nBlocks 2 --step {s} --stepmode {stepmode} --use-valid\
        --nChannels 16 --growthRate 6 --batch-size 64 --epochs 300\
        --gpu {gpu} --compress-factor {compress_factor} --scale-list {scale} --grFactor {gbf} --bnFactor {gbf} --workers 4'
    
    if evalmode is not None:
        cmd += f' --evalmode {evalmode} --evaluate-from {save_dir}/save_models/model_best.pth.tar'
    
    return cmd



import threading

class myThread (threading.Thread):
    def __init__(self, threadID, data, arch, gpuid, compress_factor, scale, stepmode, evalmode = None):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.data = data
        
        self.cmd = run_exp_cmd(scale, arch, compress_factor, stepmode, data, gpuid, evalmode)

    def run(self):
        print ("start_" + str(self.threadID))
        print(self.data)
        os.system(self.cmd) 


alpha_L = ['1-2-3-3']#,'1-2-3-3', '1-2-3']#, 1e-4, 1e-5]
c_L = [0.25]#, 0.5]#, 0.5]#, 0.05, 0.07, 0.1]
gpu_L = ['1','0']
data = ['cifar10','cifar100']
sm = ['even']#, 'lg']
arch = 'msdnetv5_ba'

_evalmode = None
#_evalmode = 'both'

t = []
g = 0

for k in range(len(data)):
    for j in range(len(c_L)):
        for i in range(len(alpha_L)):
            for b in range(len(sm)):
                _t = myThread(k, data[k], arch, gpu_L[g], compress_factor = c_L[j], scale = alpha_L[i], stepmode = sm[b], evalmode = _evalmode)
                _t.start()
                t.append(_t)
                g += 1
                if g == len(gpu_L):
                    g=0
           