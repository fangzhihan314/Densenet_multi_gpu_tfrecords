# Densenet_multi_pgu_tfrecords
Densenet in this reposite, is based on the code at "https://github.com/LaurentMazare/deep-models/tree/master/densenet" by LaurentMazare, using tfrecords format data and either single cpu or multiple gpus if possible. 

## Differences
Some differences when compared with the design in the original paper (https://arxiv.org/abs/1608.06993):
- The number of layers in each block has been set to 6 or 12 for small memory. 


## Dataset
- The used dataset for test is flowers dataset. The images are converted into standard tfrecord dataset.
- You can choose any data with tfrecord datasets, only to modify the function: read_and_decode() and inputs()
- Input image size: 224*224*3

## To do list
- Add the test step in the training. 
- Memory-efficient version
- Update the ugly code
