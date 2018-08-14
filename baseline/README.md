# gluon-reid
1.Towards Good Practices on Building Effective CNN Baseline Model for Person Re-identification

result | rank1  | ran5 | rank10 | mAP
--- | --- | --- | --- | ---
paper | 91.7 | - | - | 78.8
 |  |  |  |
ours(same as paper) | 90.9 | 96.7 | 97.7 | 76.0
ours(epoch = 20,35,40) | 90.2 | 96.4 | 97.7 | 74.4
ours(epoch = 20,35,50) | 90.2 | 96.4 | 97.5 | 74.8 
ours(epoch = 25,50,75) | 91.0 | 96.6 | 98.0 | 76.5
ours(epoch = 30,60,90) | 91.2 | 96.2 | 97.6 | 76.1
ours(epoch = 25,50,70) | 90.9 | 96.6 | 97.9 | 76.2
ours(w/o pad and crop) | 89.0 | 96.1 | 97.6 | 73.4
ours(batch size = 64, lr = 3e-4) | 90.3 | 95.8 | 97.4 | 75.4
ours(batch size = 64, lr = 3.5e-4) | 90.0 | 96.1 | 97.7 | 75.3
ours(batch size = 64, lr = 4e-4) | 89.8 | 96.0 | 97.2 | 75.7
ours(batch size = 64, lr = 5e-4) | 90.5 | 96.2 | 97.6 | 75.4
ours(batch size = 128, lr = 3.5e-4) | 88.6 | 95.6 | 96.9 | 72.7
ours(no scale and bais after bn) | 86.5 | 95.3 | 97.1 | 68.1
ours(no bais after bn) | 90.8 | 96.4 | 97.6 | 76.5
ours(no scale after bn) | 86.9 | 95.2 | 97.0 | 68.6
ours(remove downsample in last block) | 92.54 | 97.12 | 98.36 | 80.12
ours(384*128) | 90.9 | 96.2 | 97.7 | 77.2
ours(warm up, epoch = 10,30,55,80) | 90.9 | 96.5 | 97.9 | 76.5
ours(our adam) | 93.1 | 97.1 | 98.4 | 80.3
ours(our sgd, lr = 1e-1, batch size = 128, epoch = 10,25,40,50) | 92.5 | 97.4 | 98.3 | 78.2