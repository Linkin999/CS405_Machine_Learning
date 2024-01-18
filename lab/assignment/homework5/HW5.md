<h1 align = "center">HW5</h1>

<center>12011923 张旭东</center>

## Q1:

Given a set of independent observations of $\mathbf{x}$ and $\mathbf{t}$, $\mathbf{y(x,w)}$ is the output of a neural network with input vector $\mathbf{x}$ and weight vector $\mathbf{w}$, so for each pair of $\mathbf{x}$ and $\mathbf{w}$, the error is
$$
\mathbf{e}=\mathbf{t}-\mathbf{y(x,w)}
$$
 Because 
$$
p(\mathbf{t}|\mathbf{x},\mathbf{w})=\mathcal{N}(\mathbf{t}|\mathbf{y(x,w)},\mathbf{\Sigma})
$$
So
$$
p(\mathbf{e}|\mathbf{t},\mathbf{x},\mathbf{w})=\mathcal{N}(\mathbf{e}|\mathbf{0},\mathbf{\Sigma})
$$
For each pair of $\mathbf{x}$ and $\mathbf{t}$, the error function is 
$$
\begin{align}
\mathbf{e}(\mathbf{w},\mathbf{\Sigma})&=\frac{1}{\sqrt{2\pi|\mathbf{\Sigma}|}}\exp(-\mathbf{e}^{T}|\mathbf{\Sigma}^{-1}\mathbf{e})\\
&=\frac{1}{\sqrt{2\pi|\mathbf{\Sigma}|}}\exp(-(\mathbf{t-y(x,w)})^{T}\mathbf{\Sigma}^{-1}(\mathbf{t-y(x,w)}))
\end{align}
$$
The total error function is 
$$
\mathbf{E}(\mathbf{w},\mathbf{\Sigma})=\prod^{N}_{n=1}\frac{1}{\sqrt{2\pi|\mathbf{\Sigma}|}}\exp(-(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})^{T}\mathbf{\Sigma}^{-1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)}))
$$
The log likelihood function is given by
$$
\ln L(\mathbf{w},\mathbf{\Sigma})=-\frac{N}{2}(\ln |\mathbf{\Sigma}|+K\ln(2\pi))-\frac{1}{2}\sum^{N}_{n=1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})^{T}\mathbf{\Sigma}^{-1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})
$$
$K$ is the dimensionality of $\mathbf{y}$ and $\mathbf{t}$.

(a) if we assume that $\Sigma$ is fixed and known, we can drop terms that are independent of $\mathbf{w}$ from `(6)`, and by changing the sign we get the error function:
$$
E(\mathbf{w})=\frac{1}{2}\sum^{N}_{n=1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})^{T}\mathbf{\Sigma}^{-1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})
$$
(b) if we consider maximizing `(6)` w.r.t. $\Sigma$, the terms that need to be kept are 
$$
-\frac{N}{2}\ln |\mathbf{\Sigma}|-\frac{1}{2}\sum^{N}_{n=1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})^{T}\mathbf{\Sigma}^{-1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})
$$
By rewriting the second term we get
$$
-\frac{N}{2}\ln |\mathbf{\Sigma}|-\frac{1}{2}Tr[\mathbf{\Sigma}^{-1}\sum^{N}_{n=1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})^{T}]
$$
We can maximize this by setting the derivative w.r.t. $\Sigma^{-1}$ to zero, yielding 
$$
\mathbf{\Sigma}=\frac{1}{N}\sum^{N}_{n=1}(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})(\mathbf{t}_{n}-\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)})^{T}
$$
Thus the optimal value for $\Sigma$ depends on $\mathbf{w}$ through $\mathbf{y}\mathbf{(}\mathbf{x}_{n}\mathbf{,w}\mathbf{)}$.

A possible way to address this mutual dependency between $\mathbf{w}$ and $\mathbf{\Sigma}$ when it comes to optimization, is to adopt an iterative scheme, alternating between updates of $\mathbf{w}$ and $\mathbf{\Sigma}$ until some convergence criterion is reached.

## Q2: 

For a network having a logistic sigmoid output activation function
$$
y(\mathbf{x,w})=\sigma(\mathbf{w}^{T}\mathbf{x})=\frac{1}{1+\exp(\mathbf{w}^{T}\mathbf{x})}\in[0,1]
$$
For a network having an output $-1\leq y(\mathbf{x,w}) \leq 1$, the relationship between  $y_{new}(\mathbf{x,w})$ and $y_{old}(\mathbf{x,w})$ is 
$$
y_{new}(\mathbf{x,w})=2y_{old}(\mathbf{x,w})-1\in[-1,1]
$$
So
$$
\begin{align}
y_{new}(\mathbf{x,w})&=2\sigma(\mathbf{w}^{T}\mathbf{x})-1\\
&=\frac{2}{1+\exp(\mathbf{w}^{T}\mathbf{x})}-1\\
&=\frac{1-\exp(\mathbf{w}^{T}\mathbf{x})}{1+\exp(\mathbf{w}^{T}\mathbf{x})}\\
&=\frac{\exp(\frac{1}{2}\mathbf{w}^{T}\mathbf{x})-\exp(-\frac{1}{2}\mathbf{w}^{T}\mathbf{x})}{\exp(\frac{1}{2}\mathbf{w}^{T}\mathbf{x})+\exp(-\frac{1}{2}\mathbf{w}^{T}\mathbf{x})}\\
&=\tanh(\frac{1}{2}\mathbf{w}^{T}\mathbf{x})
\end{align}
$$
So the new activation function is $\tanh(\frac{a}{2})$. 

The error function for the network having a logistic sigmoid output activation function is 
$$
E(\mathbf{w})=-\sum^{N}_{n=1}[t_{n}\ln y_{n} +(1-t_{n})\ln (1-y_{n})]
$$
From the formula `(13)`, we can apply the same transformation to $y_{n}$ and $t_{n}$. So the error function of the new network is 
$$
E(\mathbf{w})=-\sum^{N}_{n=1}[\frac{t_{n}+1}{2}\ln \frac{1+y_{n}}{2} +(1-\frac{t_{n}+1}{2})\ln (1-\frac{y_{n}+1}{2})]
$$

## Q3:

For the mixture density network model, 
$$
p(\mathbf{t}|\mathbf{x})=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})\mathcal{N}(\mathbf{t}|\mathbf{\mu}_{k}(\mathbf{x}),\sigma^{2}_{k}(\mathbf{x}) )
$$

$$
E[\mathbf{t}|\mathbf{x}]=\int \mathbf{t}p(\mathbf{t}|\mathbf{x})d\mathbf{t}\\
=\int \mathbf{t}\sum^{K}_{k=1}\pi_{k}(\mathbf{x})\mathcal{N}(\mathbf{t}|\mathbf{\mu}_{k}(\mathbf{x}),\sigma^{2}_{k}(\mathbf{x}) )d\mathbf{t}\\
=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})\int \mathbf{t}\mathcal{N}(\mathbf{t}|\mathbf{\mu}_{k}(\mathbf{x}),\sigma^{2}_{k}(\mathbf{x}) )d\mathbf{t}
$$

The integral $\int \mathbf{t}\mathcal{N}(\mathbf{t}|\mathbf{\mu}_{k}(\mathbf{x}),\sigma^{2}_{k}(\mathbf{x}) )d\mathbf{t}$ just gives the mean of the Gaussian which is  $\mathbf{\mu}_{k}(\mathbf{x})$, hence
$$
E[\mathbf{t}|\mathbf{x}]=\int \mathbf{t}p(\mathbf{t}|\mathbf{x})d\mathbf{t}=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})\mathbf{\mu}_{k}(\mathbf{x})
$$
 We now introduce the shorthand notation
$$
\bar {\mathbf{t}}_{k}=\mathbf{\mu}_{k}(\mathbf{x}) \ and \ \bar {\mathbf{t}}=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})\bar t_{k}
$$

$$
\begin{align}
s^{2}(\mathbf{x})&=E[||\mathbf{t}-E[\mathbf{t}|\mathbf{x}]||^2|\mathbf{x}]\\
&=\int||\mathbf{t}-\bar {\mathbf{t}}||^{2}p(\mathbf{t}|\mathbf{x})d\mathbf{t}\\
&=\int(\mathbf{t}^T\mathbf{t}-\mathbf{t}^T\bar{\mathbf{t}}-\bar{\mathbf{t}}^T\mathbf{t}+\bar{\mathbf{t}}^T\bar{\mathbf{t}})\sum^{K}_{k=1}\pi_{k}(\mathbf{x})\mathcal{N}(\mathbf{t}|\mathbf{\mu}_{k}(\mathbf{x}),\sigma^{2}_{k}(\mathbf{x}))d\mathbf{t}\\
&=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})\int(\mathbf{t}^T\mathbf{t}-\mathbf{t}^T\bar{\mathbf{t}}-\bar{\mathbf{t}}^T\mathbf{t}+\bar{\mathbf{t}}^T\bar{\mathbf{t}})\mathcal{N}(\mathbf{t}|\mathbf{\mu}_{k}(\mathbf{x}),\sigma^{2}_{k}(\mathbf{x}) )d\mathbf{t}
\end{align}
$$

 Because
$$
E[\mathbf{x}\mathbf{x}^T]=\mathbf{\mu}\mathbf{\mu}^{T}+\mathbf{\Sigma}
$$
   So the expression `(20)` can be written as 
$$
\begin{align}
s^{2}(\mathbf{x})&=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})[\sigma^{2}_{k}+\bar{\mathbf{t}}^T\bar{\mathbf{t}}-\bar{\mathbf{t}}_{k}^T\bar{\mathbf{t}}-\bar{\mathbf{t}}^T\bar{\mathbf{t}}_{k}+\bar{\mathbf{t}}_k^T\bar{\mathbf{t}_k}]\\
&=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})[\sigma^{2}_{k}+||\bar {\mathbf{t}}_{k}-\bar {\mathbf{t}}||^2]\\
&=\sum^{K}_{k=1}\pi_{k}(\mathbf{x})[\sigma^{2}_{k}+||\mathbf{\mu}_{k}(\mathbf{x})-\sum^{K}_{l=1}\pi_{l}\mathbf{\mu}_{l}(\mathbf{x})||^2]
\end{align}
$$

## Q4: 

The logistic sigmoid function is 
$$
y(a)=\sigma(a)=\frac{1}{1+\exp(-a)}
$$
Let $b$ be the bias term, $w_1$ and $w_2$ be the weights for $\mathbf{A}$ and $\mathbf{B}$, respectively. Then, the logistic threshold unit output $y$ is given by:
$$
y=\sigma(w_1 \mathbf{A}+w_2 \mathbf{B}+b)
$$
To match the given Boolean function:

- for the input $(1,1)$, where $f(\mathbf{A},\mathbf{B})=0$, set $w_1 +w_2 +b<0$.
- for the input $(0,0)$, where $f(\mathbf{A},\mathbf{B})=0$, set $b<0$.
- for the input $(1,0)$, where $f(\mathbf{A},\mathbf{B})=1$, set $w_1+b>0$.
- for the input $(0,1)$, where $f(\mathbf{A},\mathbf{B})=0$, set $w_2 +b<0$.

The specific values of the weights can vary as long as they satisfy these conditions. An example is $b=0.5$, $w_1 =1$, $w_2=-1$.

## Q5:

(a) number of weights $=3*4*4=48$.

(b) number of ReLU operations performed on the forward pass $=3*5*5=75$.

(c) number of weights for the entire network $=48+3*5*5*4=348$.

(d) According to Universal Approximation Theorem:

let $\mathbf{C}(\mathbf{X},\R^m)$ denote the set of continuous functions from s subset $\mathbf{X}$ of a Euclidean $\R^n$. Let $\sigma \in \mathbf{C}(\R,\R)$. Note that $($ $\sigma$  $\circ$ $x$ $)_i=\sigma(x_{i})$, so  $\sigma$  $\circ$ $x$ denotes $\sigma$ applied to each component of $x$.

Then $\sigma$ is not polynomial if and only if for every $n\in\N,m\in \N$, compact $\mathbf{K}\subseteq \R^n,f \in \mathbf{C}(\mathbf{K},\R^m),\varepsilon>0$ there exist $k\in\N,\mathbf{A} \in \R^{k×n}, \mathbf{C}\in\R^{m×k}$ such that
$$
\sup||f(x)-g(x)||<\varepsilon
$$
where $g(x)=\mathbf{C}·$ $($ $\sigma$  $\circ$ $(\mathbf{A}·x+b)$ $)$.

In theory, a fully-connected neural network is a universal function approximator. 

Universal Approximation Theorem states that a neural network with a single hidden layer can approximate any continuous function, given a sufficiently large number of neurons in that layer. A fully-connected neural network involves multiple hidden layers, each with non-linear activations, enhancing its capability to represent complex classifiers. However, whether it can learn to represent a specific classifier effectively from a given set of training data is another question, which depends on factors like the complexity of the classifier, the quantity and quality of the training data, the design of the network (including the activation functions used, the learning rate, etc.), and the training algorithm used. It is also important to remember that increasing the size or complexity of a network can lead to overfitting if the network is too large relative to the amount of available training data. Overfitting is when the model learns the training data too well, to the point that it performs poorly on new, unseen data. This is why model complexity and size should be carefully managed.

(e) Fully connected neural networks and convolutional neural networks (CNNs) each have their own strengths and weaknesses, and are suited to different types of tasks. Here are some disadvantages of fully connected networks compared to CNNs, especially when dealing with image data:

1. Parameter Efficiency: Fully connected networks do not share parameters across space, leading to a large number of parameters. This can quickly become computationally expensive as the size of the inputs grows. In contrast, CNNs share parameters across space (i.e., the same filter is applied to different parts of the input), significantly reducing the number of parameters.
2. Invariance to Translations: Fully connected networks treat input features independently and don't inherently account for local spatial correlations in the input data. In contrast, CNNs, through the use of convolutional layers, automatically learn and maintain spatial hierarchies, making them better suited to tasks where spatial relationships are important, such as image and video processing. 
3. Overfitting: Due to the high number of parameters, fully connected networks are more prone to overfitting, especially when the amount of training data is small. CNNs, by having fewer parameters, can mitigate this issue to some extent.
4. Scalability: Fully connected networks do not scale well to larger images, because the number of parameters increases quadratically with the size of the input. On the other hand, CNNs handle larger images better due to parameter sharing and pooling layers, which reduce the spatial dimensions of the input.

However, it's worth noting that fully connected networks can still perform well on tasks where spatial relationships are not important, or where the input data has been sufficiently preprocessed or engineered. The choice between a fully connected network and a CNN depends on the specific task and the nature of the input data.

## Q6:

(a) The output $\mathbf{Y}$ is
$$
\begin {align} 
\mathbf{Y}&=Cw_5(Cw_1\mathbf{X_1}+Cw_3\mathbf{X_2})+Cw_6(Cw_2\mathbf{X_1}+Cw_4\mathbf{X_2})\\
&=C^2(w_1w_5\mathbf{X_1}+w_3w_5\mathbf{X_2}+w_2w_6\mathbf{X_1}+w_4w_6\mathbf{X_2})\\
&=C^{2}(w_1w_5+w_3w_5)\mathbf{X_1}+C^2(w_2w_6+w_4w_6)\mathbf{X_2}\\
&=w_{1}^{new}\mathbf{X_1}+w_{2}^{new}\mathbf{X_2}
\end {align}
$$
So:
$$
w_{1}^{new}=C^{2}(w_1w_5+w_3w_5)\\
w_{2}^{new}=C^{2}(w_2w_6+w_4w_6)
$$
(b) Yes, it is always possible to express a neural network made up of only linear units without a hidden layer, because the composition of linear functions is still a linear function, so any network of linear units can be reduced to a single-layer network.

(c)  The output $\mathbf{Y}$ is
$$
\begin {align} 
\mathbf{Y}&=t(w_5\sigma(w_1\mathbf{X_1}+w_3\mathbf{X_2})+w_6\sigma(w_2\mathbf{X_1}+w_4\mathbf{X_2}))\\
&=t(\frac{w_5}{1+\exp(w_1\mathbf{X_1}+w_3\mathbf{X_2})}+\frac{w_6}{1+\exp(w_2\mathbf{X_1}+w_4\mathbf{X_2})})
\end {align}
$$
The `XOR` of $\mathbf{X_1}$ and $\mathbf{X_2}$ for binary-valued $\mathbf{X_1}$ and $\mathbf{X_2}$ is 

| x~1~ | x~2~ |  y   |
| :--: | :--: | :--: |
|  0   |  0   |  0   |
|  1   |  0   |  1   |
|  0   |  1   |  1   |
|  1   |  1   |  0   |

Let $z_{1}=\sigma(w_1\mathbf{X_1}+w_3\mathbf{X_2}),z_{2}=\sigma(w_2\mathbf{X_1}+w_4\mathbf{X_2}),z_{3}=w_{5}z_{1}+w_{6}z_{2}$:

| X~1~ | X~2~ | z~1~       | z~2~       | z~1~+z~2~  | Y    |
| :--: | ---- | ---------- | ---------- | ---------- | ---- |
|  0   | 0    | close to 0 | close to 0 | close to 0 | 0    |
|  0   | 1    | close to 0 | close to 1 | close to 1 | 1    |
|  1   | 0    | close to 1 | close to 0 | close to 1 | 1    |
|  1   | 1    | close to 0 | close to 0 | close to 0 | 0    |

An example is $w_1=20,w,w_2=-10,w_3=-10,w_4=20,w_5=10,w_6=10$.
