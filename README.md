# Machine Learning Models Overview

This document provides a comprehensive overview of machine learning models across various categories, including both established models and emerging research topics.

## Broad Categories of Models

### Linear Models
- **Linear Regression**
- **Logistic Regression**
- **Ridge Regression**
- **Lasso Regression**
- **Elastic Net**: Combines both Ridge and Lasso regression techniques.

### Tree-Based Models
- **Decision Trees**
- **Random Forests**
- **Gradient Boosting Machines (GBM)**
- **Extreme Gradient Boosting (XGBoost)**
- **LightGBM**
- **CatBoost**
- **Oblique Decision Trees**: Trees that split on linear combinations of features.

### Support Vector Machines (SVMs)
- **SVM** (with different kernels)
- **Support Vector Regression (SVR)**
- **One-Class SVM**: Primarily used for anomaly detection.

### Neural Networks
- **Feedforward Neural Networks (FNNs)**
- **Convolutional Neural Networks (CNNs)**
- **Recurrent Neural Networks (RNNs)**
- **Long Short-Term Memory (LSTM)**
- **Gated Recurrent Units (GRU)**
- **Transformers** (e.g., BERT, GPT)
- **Autoencoders** (e.g., Variational Autoencoders)
- **Capsule Networks (CapsNets)**
- **Attention Mechanisms**: Used in RNNs, LSTMs, GRUs, and Transformers.

### Ensemble Methods
- **Bagging** (e.g., Bootstrap Aggregating)
- **Boosting** (e.g., AdaBoost)
- **Stacking**
- **Voting Classifiers**: Use majority voting or averaging for combining models.

### Clustering Algorithms
- **K-Means**
- **Hierarchical Clustering**
- **DBSCAN**
- **Mean Shift**
- **Affinity Propagation**
- **OPTICS**: Advanced clustering algorithm related to DBSCAN.

### Dimensionality Reduction
- **Principal Component Analysis (PCA)**
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
- **Linear Discriminant Analysis (LDA)**
- **Independent Component Analysis (ICA)**: Often used for signal processing.

### Bayesian Models
- **Naive Bayes**
- **Bayesian Networks**
- **Gaussian Processes**
- **Bayesian Linear Regression**
- **Markov Chain Monte Carlo (MCMC)**: A method for Bayesian inference.

### Reinforcement Learning Models
- **Q-Learning**
- **Deep Q-Networks (DQN)**
- **Policy Gradient Methods** (e.g., PPO, A3C)
- **AlphaZero**: An advanced model combining Monte Carlo Tree Search and deep learning.
- **Soft Actor-Critic (SAC)**
- **Advantage Actor-Critic (A2C/A3C)**

### Generative Models
- **Generative Adversarial Networks (GANs)**
- **Variational Autoencoders (VAEs)**
- **Normalizing Flows**: A type of generative model that models invertible mappings.
- **Diffusion Models**: Used for generating high-quality images through noise removal.

### Graph-Based Models
- **Graph Convolutional Networks (GCNs)**
- **Graph Attention Networks (GATs)**
- **Graph Isomorphism Networks (GINs)**
- **Relational Graph Convolutional Networks (RGCNs)**

### Kernel Methods
- **Kernel PCA**
- **Support Vector Machines with Kernel Functions**
- **Gaussian Processes**: Can also be considered a kernel-based method for regression and classification.

### Self-Organizing Maps (SOMs)
- **Kohonen Maps**
- **Learning Vector Quantization (LVQ)**: A supervised version of SOMs.

## Additional Types of Models

### Neuroevolutionary Models
- **NeuroEvolution of Augmenting Topologies (NEAT)**
- **Genetic Algorithms (GA)**: Often used for optimization in combination with ML models.

### Meta-Learning
- **Model-Agnostic Meta-Learning (MAML)**: Enables quick adaptation to new tasks with minimal data.
- **Prototypical Networks**: Used in few-shot learning for classification tasks with limited data.

### Anomaly Detection Models
- **Isolation Forest**: A tree-based method for anomaly detection by isolating instances.
- **One-Class SVM**: Used to find a boundary around "normal" data points for detecting outliers.

## Research and Emerging Models

### Diffusion Models
A new family of generative models that produce high-quality images by denoising random data into structured outputs.

### Neural ODEs (Ordinary Differential Equations)
Models the hidden states of neural networks as continuous functions of time using ODE solvers, gaining traction for time series and continuous data tasks.

### Graph Neural Diffusion
A combination of diffusion-based generative methods and graph neural networks, useful in drug discovery and social network analysis.

### Self-Supervised Learning Models
Methods like **SimCLR**, **BYOL**, and **MoCo** are advancing the field of learning representations without labeled data.

### Quantum Machine Learning (QML)
Quantum computing applied to machine learning tasks is still in the experimental phase but has the potential to revolutionize certain areas.

### Transformers Beyond NLP
Transformers, originally developed for NLP tasks, are being successfully applied in fields like computer vision (**Vision Transformers (ViTs)**) and time series forecasting.

### Hypernetworks
A research topic where one network generates the weights for another network, useful in few-shot learning and hyperparameter optimization.
