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

## Train
```
python main.py
```

## Test
```
python test.py
```

## Acknowledgements
This project would not have been possible without relying on some awesome repos : GMFlow, RAFT,  Swin, PyTorch. We thank
the original authors for their excellent work.
