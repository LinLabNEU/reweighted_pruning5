# Reweighted pruning

This is the submission for the MicroNet challenge on the CIFAR-100 task. 

This project is based on Pytorch. Quantization is not performed in this project.
We first introduce the pruning method for this project. Then we demonstrate how we count the parameters and operations and show the score in the end.

# Pruning method

We use reweighted L1 pruning method to prune the model. The detailed method is shown in train/reweighted_l1_pruning.pdf. The code for the pruning is in the training directory. Basically, starting from a pretrained unpruned model which achieves 81.92% accuracy on CIFAR-100, we first try to decrease the L1 norm of this model with the reweighted L1 pruning method to make model sparse. Then we set the parameters under a threshold to zero (obtain the sparsity mask) and retrain the model. Note that during retraining, the zero parameters are not updated.

To run the pruning:

```
python training/main.py
```

or refer to the training/run.sh file.

# Model

The model architecture is based on the MobileNet V2. For more details, please refer to the mobilenet_v2_cifar100_exp_30.py file and the original paper. The pruned model is saved as cifar100_mobilenetv217_retrained_acc_80.170_config_mobile_v2_0.7_threshold.pt. It can achieve 80.17% accuracy satisfying the 80% accuracy requirement.

# Verify model

To load and verify the model, run:

```
python testers.py
```
It outputs the test accuracy of the model. It also counts the number of non-zero and zero elements in the parameters. The total number of parameters is 3996704 (4.0M). Among them, there are 3107170 (3.11M) zero parameters and 889534 (0.89M) non-zero parameters. The number of bitmask is 122051 (0.1221M). So the total parameters for storage is 0.5668M (0.89M * 16 / 32 + 0.1221M) since the parameters are assumed to be 16bit if unquantized and counted as 32bit.

# Count parameters

From the output of the testers file, that is,
```
python testers.py
```
We can see that the total number of parameters is 3996704 (4.0M). Among them, there are 3107170 (3.11M) zero parameters and 889534 (0.89M) non-zero parameters. The number of bitmask is 122051 (0.1221M). So the total parameters for storage is 0.5668M (0.89M * 16 / 32 + 0.1221M). 

Parameter number: 0.5668M

# Count operations

We show how we count the operations and the operation number for scoring in the end of this part. 

We tried two ways to count the operations.

1. One way is to use the open-source pytorch-opcounter tool. It will count the number of operations during inference. To use this tool and check the performance,
```
pip install thop
python mobilenetv2.py
```
  It shows that the total number of operations is 325.4M. It counts the real operations during runtime and does not consider the sparsity since zero parameters still participate in the operations. Besides, for unquantized models, multiplication are counted as 16bit while the operation counting is based on 32bit, we believe the operation number for scoring should be smaller than the value 325.4M. We do not use this number for scoring and this number can work as a reference. 

2. We would like to use the second method to count the number of operations. It is based on the counting example from the MicroNet challenge group. ( https://github.com/google-research/google-research/blob/master/micronet_challenge/ )
The original version is for the efficientnet on tensorflow. We made necessary modifications to work for the our mobilenet_v2 model on pytorch. To run the counting,
```
  python check_model_operations.py
```
  It shows that the there are 77.84M multiplications and 153.41M additions in the case of no sparsity (setting the sparsity to 0 when print_summary). Since the multiplication is performed as 16bit and counted as 32bit, the actual number of multiplication should be 155.68M and the total number of operations is 309M (155.68M + 153.41M), which is close to and no larger than the 325.4M value in the first counting method with the tool.

So in the case of no sparsity, the total number of operations is 231.25M (77.84M+153.41M). If we consider the sparsity and set it to non-zero value, the number of operations will decrease. But since the sparsity for each layer is not the same, it is not easy to use one number to represent the sparsity of all layers. We suggest that setting the sparsity parameter to 0.5 should be an appropriate choice, considering the overall sparsity for the whole model is about 77% (1 - 0.89M/4M). By setting the sparsity parameter to 0.5, there are 39.49M multiplications and 76.7M additions according to the outputs of the check_model_operations.py file. The total operation number is 116.19M. The real operation number should be smaller than this, because most of the layers have a sparsity larger than 0.5 and the overall sparsity of the whole model is about 0.77. But we think we can use this operation number in scoring.

operation number: 116.19M

# Score 

For CIFAR-100, parameter storage and compute requirements will be normalized relative to WideResNet-28-10, which has 36.5M parameters and 10.49B math operations.

So the score is 0.5668M / 36.5M + 116.19M / 10.49B = 0.0266

# Team member

The team name is Woody.

This is an collaboration of Northeastern University, Indiana University and IBM corporation. The team members are listed as follows, 
- Northeastern University
  - Pu Zhao
  - Zheng Zhan
  - Zhengang Li
  - Xiaolong Ma
  - Yanzhi Wang
  - Xue Lin
- Indiana University
  - Qian Lou
  - Lei Jiang
- MIT-IBM Watson AI Lab, IBM Research
  - Gaoyuan Zhang
  - Sijia Liu

contact: zhao.pu@husky.neu.edu or zhan.zhe@husky.neu.edu or xue.lin@northeastern.edu
