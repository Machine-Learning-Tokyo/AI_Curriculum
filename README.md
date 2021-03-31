# AI Curriculum
Open Deep Learning and Reinforcement Learning lectures from top Universities like Stanford University, MIT, UC Berkeley.

<!-- toc -->

## Contents

- [Applied Machine Learning](#applied-machine-learning)
  - Cornell CS5785: Applied Machine Learning | Fall 2020
- [Introduction to Deep Learning](#introduction-to-deep-learning)
  - UC Berkeley CS 182: Deep Learning | Spring 2021
  - MIT 6.S191: Introduction to Deep Learning | 2020
- [CNNs for Visual Recognition](#cnns-for-visual-recognition)
  - CS231n: CNNs for Visual Recognition, Stanford | Spring 2019
- [NLP with Deep Learning](#nlp-with-deep-learning)
  - CS224n: NLP with Deep Learning, Stanford | Winter 2019
- [Unsupervised Learning](#unsupervised-learning)
  - CS294-158-SP20: Deep Unsupervised Learning, UC Berkeley | Spring 2020
- [Multi-Task and Meta Learning](#multi-task-and-meta-learning)
  - Stanford CS330: Multi-Task and Meta Learning | 2019
- [Deep Learning (with Pytorch)](#deep-learning)
  - DS-GA 1008: Deep Learning | Spring 2020
- [Deep Reinforcement Learning](#deep-reinforcement-learning)
  - CS285: Deep Reinforcement Learning, UC Berkeley | Fall 2020



# Machine Learning

## Applied Machine Learning

### Cornell CS5785: Applied Machine Learning | Fall 2020

Lecture videos and resources from the Applied Machine Learning course at Cornell Tech, taught in Fall 2020. The lectures are covering ML from the very basics, including the important ML algorithms and how to apply them in practice. All materials are executable, the slides are Jupyter notebooks with programmatically generated figures. Readers can tweak parameters and regenerate the figures themselves.

- [Lecture series, Youtube](https://www.youtube.com/playlist?list=PL2UML_KCiC0UlY7iCQDSiGDMovaupqc83)
- [GitHub notebooks](https://github.com/kuleshov/cornell-cs5785-applied-ml)

Source: Cornell

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/cornell_applied_ml.png" width="800"></p>](https://www.youtube.com/playlist?list=PL2UML_KCiC0UlY7iCQDSiGDMovaupqc83)


# Deep Learning

## Introduction to Deep Learning

### UC Berkeley CS 182: Deep Learning | Spring 2021

- [Website](https://cs182sp21.github.io/)
- [Lecture series](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A)

Source: Berkeley

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/deep_learning_berkeley.png" width="800"></p>](https://www.youtube.com/playlist?list=PL_iWQOsE6TfVmKkQHucjPAoRtIJYt8a5A)


### MIT 6.S191: Introduction to Deep Learning | 2020
[Lecture Series](http://introtodeeplearning.com/)

MIT's introductory course on deep learning methods with applications to computer vision, natural language processing, biology, and more! Students will gain foundational knowledge of deep learning algorithms and get practical experience in building neural networks in TensorFlow. Course concludes with a project proposal competition with feedback from staff and panel of industry sponsors. Prerequisites assume calculus (i.e. taking derivatives) and linear algebra (i.e. matrix multiplication), we'll try to explain everything else along the way! Experience in Python is helpful but not necessary. Listeners are welcome!

Source: MIT

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/mit.gif" width="600"></p>](http://introtodeeplearning.com/)

# CNNs for Visual Recognition

### CS231n: CNNs for Visual Recognition, Stanford | Spring 2019
[Lecture Series](http://cs231n.stanford.edu)

Computer Vision has become ubiquitous in our society, with applications in search, image understanding, apps, mapping, medicine, drones, and self-driving cars. Core to many of these applications are visual recognition tasks such as image classification, localization and detection. Recent developments in neural network (aka “deep learning”) approaches have greatly advanced the performance of these state-of-the-art visual recognition systems. This course is a deep dive into details of the deep learning architectures with a focus on learning end-to-end models for these tasks, particularly image classification. During the 10-week course, students will learn to implement, train and debug their own neural networks and gain a detailed understanding of cutting-edge research in computer vision. The final assignment will involve training a multi-million parameter convolutional neural network and applying it on the largest image classification dataset (ImageNet). We will focus on teaching how to set up the problem of image recognition, the learning algorithms (e.g. backpropagation), practical engineering tricks for training and fine-tuning the networks and guide the students through hands-on assignments and a final course project. Much of the background and materials of this course will be drawn from the ImageNet Challenge.

Source: Stanford

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/stanford.png" width="600"></p>](http://cs231n.stanford.edu)

# NLP with Deep Learning

### CS224n: NLP with Deep Learning, Stanford | Winter 2019
[Lecture Series](http://web.stanford.edu/class/cs224n/index.html#schedule)

Natural language processing (NLP) is a crucial part of artificial intelligence (AI), modeling how people share information. In recent years, deep learning approaches have obtained very high performance on many NLP tasks. In this course, students gain a thorough introduction to cutting-edge neural networks for NLP.

Source: Stanford

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/nlp_stanford.png" width="600"></p>](http://web.stanford.edu/class/cs224n/index.html#schedule)

# Unsupervised Learning

### CS294-158-SP20: Deep Unsupervised Learning, UC Berkeley | Spring 2020
[Lecture Series, YouTube](https://www.youtube.com/playlist?list=PLwRJQ4m4UJjPiJP3691u-qWwPGVKzSlNP)

This course covers two areas of deep learning in which labeled data is not required: Deep Generative Models and Self-supervised Learning. Recent advances in generative models have made it possible to realistically model high-dimensional raw data such as natural images, audio waveforms and text corpora. Strides in self-supervised learning have started to close the gap between supervised representation learning and unsupervised representation learning in terms of fine-tuning to unseen tasks. This course will cover the theoretical foundations of these topics as well as their newly enabled applications.

- Source: [UC Berkeley](https://sites.google.com/view/berkeley-cs294-158-sp20/home)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/deep_unsupervised.png" width="600"></p>](https://sites.google.com/view/berkeley-cs294-158-sp20/home)


# Multi-Task and Meta Learning

### Stanford CS330: Multi-Task and Meta Learning | 2019
[Lecture Series, YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5)

While deep learning has achieved remarkable success in supervised and reinforcement learning problems, such as image classification, speech recognition, and game playing, these models are, to a large degree, specialized for the single task they are trained for. This course will cover the setting where there are multiple tasks to be solved, and study how the structure arising from multiple tasks can be leveraged to learn more efficiently or effectively. This includes:

- goal-conditioned reinforcement learning techniques that leverage the structure of the provided goal space to learn many tasks significantly faster
- meta-learning methods that aim to learn efficient learning algorithms that can learn new tasks quickly
- curriculum and lifelong learning, where the problem requires learning a sequence of tasks, leveraging their shared structure to enable knowledge transfer

Source: [Stanford University CS330](http://cs330.stanford.edu/)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/meta_learning.png" width="600"></p>](http://cs330.stanford.edu/)

### DS-GA 1008: Deep Learning | SPRING 2020
[Lecture Series](https://atcold.github.io/pytorch-Deep-Learning/)

This course concerns the latest techniques in deep learning and representation learning, focusing on supervised and unsupervised deep learning, embedding methods, metric learning, convolutional and recurrent nets, with applications to computer vision, natural language understanding, and speech recognition. The lectures are taught by Yann LeCun and the entire lecture series is also available as a YouTube Playlist officially provided by them at [https://www.youtube.com/playlist?list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq](https://www.youtube.com/playlist?list=PLLHTzKZzVU9eaEyErdV26ikyolxOsz6mq).

Source: NYU Center for Data Science

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/nyu.png" width="800"></p>](https://atcold.github.io/pytorch-Deep-Learning/)

# Deep Reinforcement Learning

### CS285: Deep Reinforcement Learning, UC Berkeley | Fall 2020
[Lecture Series, YouTube](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc)

[<p align="center"><img src="https://github.com/Machine-Learning-Tokyo/AI_Curriculum/blob/master/images/levine.png" width="800"></p>](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc)

