# FocoNet

This is a transformer-based neural network to solve for the focal mechanism by combining the first-motion polarities, S/P amplitude ratios, SNRs from a set of stations.

## FocoNet structure and the workflow to deterine a focal mechanism:

![FOCONETnew](https://github.com/user-attachments/assets/b43ad37e-f878-4d80-9a97-b24d3006dd82)

## FocoNet Versions:

* FocoNet_Full: the FocoNet model using the first motion polarities, the S/P amplitude ratios (both the max{S}/|P| and the S/P ratios from different channels), and the signal to noise ratios (SNRs) to determine focal mechanisms.

* FocoNet_SP: the FocoNet model using only the first motion polarities and the max{S}/|P| ratios to determine focal mechanisms.
  
* FocoNet_O: the FocoNet model using only the first motion polarities to determine focal mechanisms.

They are only different in dataloaders (data_loader.py), dimensions of the input layer (model.py), and model checkpoints (model/FocoNet_x.pth)

They are trained and tested on the same datasets.

## Package Requirement

* numpy < 2
* pytorch
* pyrocko
* obspy

Training requires cuda environment.

To create a conda environment, follow the commands below:

```
conda create -n foconet python=3.12

pip3 install numpy==1.26
pip3 install torch
pip3 install pyrocko
pip3 install obspy

conda activate foconet
```

## Train FocoNet

Please note that the training procedure is only tested successful on a linux (ubuntu 24.0) system

1. Download the Training set and Test set:

    https://zenodo.org/records/16938136?preview=1
   
3. Copy the Training and Test sets to the following paths:

    ```
    FocoNet_Full/Train/Train_set.npy
    FocoNet_Full/Test/Test_set.npy
    ```
    (our similarily, in the FocoNet_O and FocoNet_SP folder)

4. Confirm Training parameters

    Open file
    ```
    FocoNet/config.py
    ```
    Confirm or make necessary changes to the path of the Training and Test sets. 
    Confirm or make necessary changes to the batchsize, learning rate, and training epochs.

5. Run main.py

    ```
    python3 main.py [Optional] --mode='TrainFromStart'
    ```

## Evaluate FocoNet (Make Predictions)

The evaluation procedure is tested successful on both linux and macOS systems.

1. The model files (checkpoints) should have been included in the folder
   
    ```
    FocoNet_Full/model/FocoNet_Full.pth
    ```
    (our similarily, in the FocoNet_O and FocoNet_SP folder)

2. Download the Evaluation sets:

    https://zenodo.org/records/16938136?preview=1

3. Put the Evaluation sets under the following path, and here we list all of our Evaluation sets in our paper:

   Synthetic Dataset:
   | Evaluation Name	| File Path | Event Number	| Azimuth Coverage	| Station Number	| Velocity Model	| Polarity Accuracy |
   | ----	|  ---- |  ----	|  ----	|  ----	|  ----	|  ---- |
   |Evaluation Set 12-STA|	```Dev_Validation/Atest_12stations.npy``` |	512 |	360º |	12	|Amatrice	|100% accurate|
   |Evaluation Set 24-STA|	```Dev_Validation/Atest_24stations.npy``` |	512 |	360º |	24	|Amatrice	|100% accurate|
   |Evaluation Set 32-STA|	```Dev_Validation/Atest_32stations.npy``` |	512 |	360º |	32	|Amatrice	|100% accurate|
   |Evaluation Set 12-STA-f|	```Dev_Validation/Btest_12stations.npy``` |	512 |	360º|	12	|Amatrice	|12%-flipped|
   |Evaluation Set 24-STA-f|	```Dev_Validation/Btest_24stations.npy``` |	512 |	360º|	24	|Amatrice	|12%-flipped|
   |Evaluation Set 32-STA-f|	```Dev_Validation/Btest_32stations.npy``` |	512 |	360º|	32	|Amatrice	|12%-flipped|
   |Evaluation Set 090-AZI|	```Dev_Validation/Atest_090AziGap.npy``` |	512 |	 90º |	24	|Amatrice	|100% accurate|
   |Evaluation Set 180-AZI|	```Dev_Validation/Atest_180AziGap.npy``` |	512 |	180º |	24	|Amatrice	|100% accurate|
   |Evaluation Set 270-AZI|	```Dev_Validation/Atest_270AziGap.npy``` |	512 |	270º |	24	|Amatrice	|100% accurate|
   |Evaluation Set 090-AZI-f|	```Dev_Validation/Btest_090AziGap.npy``` |	512 |	 90º |	24	|Amatrice	|12%-flipped|
   |Evaluation Set 180-AZI-f|	```Dev_Validation/Btest_180AziGap.npy``` |	512 |	180º |	24	|Amatrice	|12%-flipped|
   |Evaluation Set 270-AZI-f|	```Dev_Validation/Btest_270AziGap.npy``` |	512 |	270º |	24	|Amatrice	|12%-flipped|
   |Evaluation Set 12-STA-sc|	```Dev_SouthCal/Atest_12stations.npy``` |	512 |	360º |	12	|Amatrice	|100% accurate|
   |Evaluation Set 24-STA-sc|	```Dev_SouthCal/Atest_24stations.npy``` |	512 |	360º |	24	|Amatrice	|100% accurate|
   |Evaluation Set 32-STA-sc|	```Dev_SouthCal/Atest_32stations.npy``` |	512 |	360º |	32	|Amatrice	|100% accurate|
   |Evaluation Set 12-STA-scf|```Dev_Validation/Btest_12stations.npy``` |	512 |	360º|	12	|Amatrice	|12%-flipped|
   |Evaluation Set 24-STA-scf|```Dev_Validation/Btest_24stations.npy``` |	512 |	360º|	24	|Amatrice	|12%-flipped|
   |Evaluation Set 32-STA-scf|```Dev_Validation/Btest_32stations.npy``` |	512 |	360º|	32	|Amatrice	|12%-flipped|
   
   Real Datasets:
   | Evaluation Name	| File Path | Event Number	| Azimuth Coverage	| Station Number	| Velocity Model	| Polarity Accuracy |
   | ----	|  ---- |  ----	|  ----	|  ----	|  ----	|  ---- |
   |Evaluation Set 12-STA-r|	```Dev_Amatrice_Standard/Amatrice_Standard_12.npy``` |	628 |	~360º|	12	| -	|~12%-flipped|
   |Evaluation Set 24-STA-r|	```Dev_Amatrice_Standard/Amatrice_Standard_12.npy``` |	628 |	~360º|	24	| -	|~12%-flipped|
   |Evaluation Set 32-STA-r|	```Dev_Amatrice_Standard/Amatrice_Standard_12.npy``` |	628 |	~360º|	32	| -	|~12%-flipped|
   |Evaluation Set A-Class|	```Dev_Amatrice/Amatrice_A_set_M.npy``` |	4452 |	~360º|	15-32	| -	| - |
   |Evaluation Set D-Class|	```Dev_Amatrice/Amatrice_D_set_M.npy``` |	1101 |	~360º|	18-32	| -	| - |

   Additional Datasets:
   
   ```Dev_Amatrice/Amatrice_A_set.npy```: All the Amatrice earthquakes with A-Class SKHASH solutions (magnitude can be smaller than 1.5)
   
   ```Dev_Amatrice/Amatrice_B_set.npy```: All the Amatrice earthquakes with B-Class SKHASH solutions (magnitude can be smaller than 1.5)
   
   ```Dev_Amatrice/Amatrice_C_set.npy```: All the Amatrice earthquakes with C-Class SKHASH solutions (magnitude can be smaller than 1.5)
   
   ```Dev_Amatrice/Amatrice_D_set.npy```: All the Amatrice earthquakes with D-Class SKHASH solutions (magnitude can be smaller than 1.5)
   
   ```Dev_Amatrice/Amatrice_All_set.npy```: All the 400,000+ Amatrice earthquakes (Meier et al., 2025)
   
   Additional Notes:

   Each evaluation set is a dictionary containing the event info, seismic input (polarities, S/P ratios, SNRs), and station locations. The users can read and check these dictionaries in python through:
   ```python
   data = np.load('Dev_Validation/Atest_24stations.npy', allow_pickle='True').item()
   ```

   Evaluation Set 63-STA-f is not included as it is only applied to SKHASH. We haven't made a FocoNet readable .npy file for it.

5. Confirm Evaluation paths

    Open file
    ```
    FocoNet/config.py
    ```
    Confirm or make necessary changes to the path of the Evaluation sets.

6. Run main.py

    ```
    python3 main.py --mode='Eval'
    ```
    
7. Check result

    Results are saved to:
    ```
    dev_outputs.npy
    ```
    as a $N \times 9$ matrix which defines the three normalized vectors for the three axis for each focal mechanism.
   
    Each row are (Px, Py, Pz, Tx, Ty, Tz, Bx, By, Bz).
   
    The reference results (labels) are saved to:
    ```
    dev_referece.npy
    ```
    in the same format with the predictions

    In the near future we will update the output format to strike/dip/rake.
   
    Currently, users can refer to our jupyter notebook *check_result.ipynb* to convert the PTB axis to focal mechanisms using pyrocko package.

## Citation:
Song, X., Meier, M., Ellsworth, W. L., & Beroza, G. C., FocoNet: transformer-based focal-mechanism determination, JGR Machine Learning in review, 2025

## References:
Meier, M.-A., Lanza, F., and Martinez-Garzon, P., A deep catalogue of 56k focal mechanisms for the 2016 Amatrice, Italy earthquake sequence, BSSA in review, 2025
