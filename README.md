# DECFusion
Code of DECFusion: DECFusion: A Lightweight Infrared and Visible Image Fusion Method Based on Retinex Decomposition

## Tips:<br>
Due to file size issues, the training set has been removed from the code and the MSRS dataset can be downloaded here: https://github.com/Linfeng-Tang/MSRS
Place the downloaded training dataset under: dataset/MSRS/ path.

## To Train
Run "python main.py" to train the model.
The training data are selected from the MSRS dataset. 

## To Test
Run "python evalFuse.py" to test the model.
The images generated by the test will be placed under the result/Fuse path.

If this work is helpful to you, please cite it as:
```
@article{
  title={ DECFusion: A Lightweight Infrared and Visible  Image Fusion Method Based on Retinex Decomposition},
author={Quanquan Xiao ,Haiyan jin,Haonan Su,etc},
}
```
If you have any question, please email to me (1211211001@stu.xaut.edu.cn).

