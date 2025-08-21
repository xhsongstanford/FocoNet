# FocoNet

This is a transformer-based neural network to solve for the focal mechanism by combining the first-motion polarities, S/P amplitude ratios, SNRs from a set of stations.

## FocoNet structure and the workflow to deterine a focal mechanism:

![FOCONETnew](https://github.com/user-attachments/assets/b43ad37e-f878-4d80-9a97-b24d3006dd82)


## Package Requirement

* numpy
* pytorch
* pyrocko

Training requires cuda environment

## Train FocoNet

1. Download the Training set and Test set:


   
2. Copy the Training and Test sets to the following paths:

    ```
    FocoNet/Train/Train_set.npy
    FocoNet/Test/Test_set.npy
    ```

3. Confirm training parameters

    Open file
    ```
    FocoNet/config.py
    ```
    Confirm or make necessary changes to the path of the Training and Test sets. 
    Confirm or make necessary changes to the batchsize, learning rate, and training epochs.

5. Run main.py

    ```
    python3 main.py [Optional] --mode='Train'
    ```

## Evaluate FocoNet (Make Predictions)

1. Download the model checkpoints:

2. Copy checkpoint the following path:

    ```
    FocoNet/model/FocoNet_ckpt.pth
    ```

## Citation:
Song, X., Meier, M., Ellsworth, W. L., & Beroza, G. C. (2024, 09). Transformer-based joint-station seismological analysis applied to focal-mechanism determination.. Poster Presentation at 2024 SCEC Annual Meeting.

## References:
Meier, M.-A., Lanza, F., and Martinez-Garzon, P.: A deep catalogue of 56k focal mechanisms for the 2016 Amatrice, Italy earthquake sequence, EGU General Assembly 2023, Vienna, Austria, 24–28 Apr 2023, EGU23-7167
