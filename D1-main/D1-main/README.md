# Task D1: Fine-grained classification
**8 Marks**

Given an image, predict its corresponding fine-grained class label. Formally, given the pre-
trained backbone $f$ (i.e. the CNN we provide, which does not have a classification layer),
you need to design a classifier $h$ (e.g. a linear layer or a multi-layer perceptron) that takes
the backbone’s image embedding in input and outputs a class prediction, i.e. $\hat{y}^f = h\big(f(x)\big)$.
You will need to train both $h$ and $f$ using the images and labels in the training set $D^\text{train}$.
For testing, you will need to use the images in the test set $D^\text{test}$, without feeding the ground
truth label $y^f$ to the model. The model will be evaluated by calculating the top-1 accuracy
between the predicted class $\hat{y}^f$ and the ground truth label $y^f$ for all images in the test set.
In your report, discuss the design choices for your classifier and the performance of your
model on the test set.

## Usage of Libraries / Restrictions

- You can only use PyTorch, NumPy and Matplotlib. Any other library is not
allowed.
- You cannot use the `torchvision.datasets.CIFAR100` class since we will use a custom subsample of the dataset. You will need to implement the `Dataset` class yourself.
- You can only use the model we prepared (with the specified pre-training weights).
- You must implement the classifiers yourself using classes from the `torch.nn` module. You can use existing loss functions to train your model.

> For any library function you use, please ensure you carefully read and understand its documentation. Note that some functions may behave differently depending on the operating system. Specific restrictions and permissions for each tasks are detailed below. Using external libraries other than the specified ones will cause issues when testing your code from our side.

## Marking Criteria

An explanation of the architecture choice of your classifier and how your choices affected the performance of the model.

For example:
- Did having few or many layers result in a better performance?
- Did the hidden dimension of the linear layers affect performance?
  - How about Dropout?
- What loss function and optimiser did you use, and why?
  - If you tried more than one, did they differ in performance?
- How did hyper-parameters (e.g. learning rate, batch size) affect your results?
- Comparing tasks D1, D2, and D3, which one was easier/harder, which one got the best/poorest results, and why?
- Did you use the same classifier architecture for all three tasks?

> Note that the report is the most important part of your assessment. We will run your code and check your output on a private test set with the exact same structure of the public test set. The performance of your output will influence your final mark, but we will not mark the quality of your code.

> We will publish our results on the public test set so that you can gauge whether your models achieve decent results. Your results must pass a minimum threshold for each task, but there are no extra marks for any extra points your model scores compared to our implementation. In fact, please bear in mind that we care about the explanation of what you have done and the interpretation of the results more than the number themselves. This applies especially for tasks D1-D7: we know that getting good results with a CNN requires extensive training and fine-tuning. This is why we have chosen a small CNN and prepared a small subsample of the dataset to make training on an average CPU bearable (nevertheless, we recommend using [Colab](https://colab.google) with GPU acceleration to speed things up).

| Criterion | Description |
|----|----|
| **Explanation of approach** | The method is explained well, and it clearly details how the various steps are carried out. |
| **Analysis of performance** | The performance is analysed and evaluated well with both quantitative and qualitative evaluation. |
| **Analysis of parameters** | The sensitivity of the task with respect to relevant parameters associated with the algorithm are analysed. |
| **Code output** | The code runs without errors and achieves good results on a held-out test set. Results will be compared with our own implementation. |

## Dataset

In this part of the coursework you will use a pre-trained CNN to perform various tasks. The tasks are based on the CIFAR 100 dataset [^1], which contains 100 classes, with 600 images for each class. Classes are clustered in superclasses, i.e. we have a coarse-grained label (e.g. “flowers”) attached to several fine-grained labels (e.g. “orchids, poppies, roses”). 

Table 1 illustrates the 100 classes grouped into superclasses:

|Superclass|Classes|
|-|-|
|aquatic mammals|beaver, dolphin, otter, seal, whale|
|fish aquarium|fish, flatfish, ray, shark, trout|
|flowers|orchids, poppies, roses, sunflowers, tulips|
|food containers|bottles, bowls, cans, cups, plates|
|fruit and vegetables|apples, mushrooms, oranges, pears, sweet peppers|
|household electrical devices|clock, computer keyboard, lamp, telephone, television|
|household furniture|bed, chair, couch, table, wardrobe|
|insects|bee, beetle, butterfly, caterpillar, cockroach|
|large carnivores|bear, leopard, lion, tiger, wolf|
|large man-made outdoor things|bridge, castle, house, road, skyscraper|
|large natural outdoor scenes|cloud, forest, mountain, plain, sea|
|large omnivores and herbivores|camel, cattle, chimpanzee, elephant, kangaroo|
|medium-sized mammals|fox, porcupine, possum, raccoon, skunk|
|non-insect invertebrates|crab, lobster, snail, spider, worm|
|people|baby, boy, girl, man, woman|
|reptiles|crocodile, dinosaur, lizard, snake, turtle|
|small mammals|hamster, mouse, rabbit, shrew, squirrel|
|trees|maple, oak, palm, pine, willow|
|vehicles 1|bicycle, bus, motorcycle, pickup truck, train|
|vehicles 2|lawn-mower, rocket, streetcar, tank, tractor|

Figure 4 shows a few samples from the dataset:

<figure>
    <img width="1271" height="988" alt="006" src="https://github.com/user-attachments/assets/5bb746a2-d251-4194-8c13-572554c245cf" />
  <figcaption><strong>Figure 4</strong>: Samples from CIFAR 100. Image from <a href="https://www.cs.toronto.edu/~kriz/cifar.html">the CIFAR 100 website</a>.
  </figcaption>
</figure>

<br/>
<br/>

We provide a sub-sampled version of the dataset, split in training and test sets. The training set contains 100 samples per fine-grained class (10,000 images in total) and the test contains 25 samples per fine-grained class (2,500 images in total). The test set used for marking is private and different from the one you will receive (but it is of the same size)

## Model

You will use the MobileNetV3 [^2] model pre-trained on ImageNet [^3]. You are not allowed to use any other model or pre-training. You will use the model’s backbone (i.e. the core of the model before the classification layer) and will have to build your own classifiers. You will have to fine-tune the model (backbone and classifier). You will need to keep the versions of the model that you trained in each task to complete some other tasks. Note that this is a tiny CNN designed for efficiency (it can run on consumer CPUs). We chose this model to make training fast but naturally the size of the model will bring some limitations. Keep this in mind when running your experiments and writing your report.

## Notation

The notation for the D1-D7 tasks is defined in Table 2:

|Symbol|What it is|
|------|----------|
|$x$|image|
|$y^f$|fine-grained class (e.g. “orchids”)|
|$y^c$|coarse-grained class (e.g. “flower”)|
|$D^\text{train} = \Big\\{\big(x_i, y^f_i , y^c_i \big), i \in \big\\{1, ... , N_\text{train}\big\\}\Big\\}$|Training set consisting of $N_\text{train}$ samples|
|$D^\text{train} = \Big\\{\big(x_i, y^f_i , y^c_i \big), i \in \big\\{1, ... , N_\text{train}\big\\}\Big\\}$|Test set consisting of Ntest samples|
|$f$|CNN backbone (before the classification output)|
|$h$|classifier (e.g. linear layer or multilayer perceptron)|


## References

[^1]: Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.
[^2]: Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, et al. Searching for MobileNetV3. In _International Conference on Computer Vision_, 2019.
[^3]: Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. ImageNet: A large-scale
hierarchical image database. In _Conference on Computer Vision and Pattern Recognition_, 2009.
