# R3DCNN-tensorflow
A tensorflow implementation of the R3DCNN network by Molchanov (2016).

This is directly taken out of my research project and probably currently somewhat unusable for general purposes. Feel free to make the appropiate changes!

The network is based on the C3D network by Tran. The C3D implementation is taken from [hx173149](https://github.com/hx173149/C3D-tensorflow).

I was using a configuration file to describe the dataset which is used in the config file.
This looks then in the folder of the config file for folders named like the labels.
Every file in there is labelled with the folder name.

It contains a function to pretrain a C3D network.

Shortcomings:
This currently **does not support** multiple GPUs.
The video length is fixed.
The preprocessing is not dynamic:
    - Image size is fixed to 112x112
    - works only for 1-channel (depth) video
- Overall there might be some unnecessary code.

I hope this is still helpful to someone.
