# E:\zfx\codes\DxFormer\saved\dxy\dense_all
from utils import load_pickle
dxy = load_pickle('dxy/tra_dense_all.pt')
mz4 = load_pickle('mz4/tra_dense_all.pt')
mz10 = load_pickle('mz10/tra_dense_all.pt')

# for loop
def loop_read(data,name):
    for i in data:
        print(i)
        # 将i输出到temp.txt中
        with open(f'{name}_last.txt', 'a') as f:
            f.write(str(i)+'\n')
loop_read(mz4)
