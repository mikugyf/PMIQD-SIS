## PMIDB-SIS

### Downloading the PMIDB-SIS Dataset
PMIDB-SIS Dataset including images, subjective scores and eye movement data.

[Baiduyun](https://pan.baidu.com/s/1yY_Xd3cm2l8DFJzijrlm5w) (password: wkm5) 

[GoogleDrive](https://drive.google.com/file/d/1PzoSv5F7FBP-8HTEf5t3nwAGli_3ARTH/view?usp=sharing)


### Requirement
```
pytorch
numpy
PIL
scipy
h5py
```
### Train
Use the main.py and set the parameters.
```python
python main.py --datamat='./data/PMIQD-SIS.mat' --imgset='./data/dataset/PMIQD-SIS' --batch_size=4
```
### Test
Test on the datasets.
```python
python evaluate.py --imgdir='/testimg'
```
