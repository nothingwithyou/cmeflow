CME: UTFlowNet
====
### Paper
## Install
```
git clone https://github.com/nothingwithyou/cmeflow.git
```
The software and versions I used are listed in requirements.txt. It is not necessary to strictly follow them.
Tested with torch>=2.0.1, python>=3.8. For installation, please refer to the [pytorch](https://pytorch.org/) official website

## Preparation
You need to create `img1_name_list.json` and `img2_name_list.json` files for your dataset, which store the file paths for 
the first and second images, respectively. The files are stored in .pt format (for efficient reading by the computer, 
use `torch.save` when saving). If other formats are needed, please modify the reading code in the _**data/datasets**_ folder.

If you need help, feel free to contact me with email [cqy@swfu.edu.cn](mailto:cqy@swfu.edu.cn) or 
[1293881481@qq.com](mailto:cqy@swfu.edu.cn) (Common mailbox).

Here is my code for making the json for reference (**_datapath/events/imagefiles_**):
```python
import json
from glob import glob
data_dir = glob(data_path + "/*")
img1_name_list = []
img2_name_list = []
for dd in data_dir:
    if '.json' not in dd:
        a = glob(dd + '/*.pt')
        a.sort()
        img1_name_list.extend(a[:-1])
        img2_name_list.extend(a[1:])
json.dump(img1_name_list, open("img1_name_list.json",'w'))
json.dump(img2_name_list, open("img2_name_list.json",'w'))
```
If you are using the version that mistakenly uploaded the unpublished work for multi-frame moving window dataset, 
you can generate the window json as follows
```python
import json
from glob import glob
img1_name_list = []
img2_name_list = []
sequence_length = 5
data_dir = glob.glob(data_path + "/*")
for dd in data_dir:
    a = glob(dd + '/*.pt')
    a.sort()
    input_seqs_1 = []
    input_seqs_2 = []
    for i in range(len(a)-sequence_length -1):
        input_seqs_1.append(a[i:i+sequence_length])
        input_seqs_2.append(a[i+1:i+1+sequence_length])
    img1_name_list.extend(input_seqs_1)
    img2_name_list.extend(input_seqs_2)
json.dump(img1_name_list, open(data_path + "/img1_name_list.json",'w'))
json.dump(img2_name_list, open(data_path + "/img2_name_list.json",'w'))
```
You need to change the parameters `--data_min` and `--data_max` in **_main.py_** according to the maximum and minimum of your own dataset
## Train
Our preprocessing is designed to ensure the original accuracy of the data, and we do less customized processing. 
If you can provide better data preprocessing methods yourself, it will improve the results of the network.

If you want better accuracy, we recommend using a larger resolution **512*512**. This means more GPU memory or a smaller 
batch size is needed.
```
python main.py
```
## Test
This is unsupervised training. We recommend that you train and reason based on your own dataset. We will also provide
our training dataset and pre-trained weights later.

```
python test.py
```

## Acknowledgements
This project would not have been possible without relying on some awesome repos : GMFlow, RAFT,  Swin, PyTorch. We thank
the original authors for their excellent work.
