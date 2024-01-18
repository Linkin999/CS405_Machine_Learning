<h1 align = "center">HW5</h1>

<center>12011923 张旭东</center>

## Q1：

To show that
$$
\mathbf{\scriptsize{W}}\varpropto \mathbf{m_{2}-m_{1}}
$$
Starting with the class separation criterion 
$$
m_{2}-m_{1}=\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}(\mathbf{m_{2}-m_{1}})
$$
With the constraint $\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\mathbf{\scriptsize{W}}=\mathbf{1}$, the Lagrangian for this optimization problem  is given by 
$$
L(\mathbf{\scriptsize{W}},\lambda)=\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}(\mathbf{m_{2}-m_{1}})-\lambda(\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\mathbf{\scriptsize{W}}-\mathbf{1})
$$
Taking the derivative of $L$ with respect to $\mathbf{\scriptsize{W}}$ and setting it to zero, we can get:
$$
\frac{\partial L  }{\partial \mathbf{\scriptsize{W}}}=(\mathbf{m_{2}-m_{1}})-2\lambda \mathbf{\scriptsize{W}}=0
$$
Then we can get:
$$
\mathbf{\scriptsize{W}}=\frac{\mathbf{m_{2}-m_{1}}}{2\lambda}
$$
The constraint $\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\mathbf{\scriptsize{W}}=\mathbf{1}$ implies that
$$
(\frac{\mathbf{m_{2}-m_{1}}}{2\lambda})^{\mathbf{T}}(\frac{\mathbf{m_{2}-m_{1}}}{2\lambda})=1
$$

$$
({\mathbf{m_{2}-m_{1}}})^{\mathbf{T}}({\mathbf{m_{2}-m_{1}}})=4\lambda^{2}
$$

Then we can get the expression of $\lambda$:
$$
\lambda=\frac{\pm\sqrt{({\mathbf{m_{2}-m_{1}}})^{\mathbf{T}}({\mathbf{m_{2}-m_{1}}})}}{2}
$$
Substituting $\lambda$ back in to the expression for $\mathbf{\scriptsize{W}}$:
$$
\mathbf{\scriptsize{W}}=\frac{\mathbf{m_{2}-m_{1}}}{\pm\sqrt{({\mathbf{m_{2}-m_{1}}})^{\mathbf{T}}({\mathbf{m_{2}-m_{1}}})}}
$$
The $\pm$ sign indicates that the direction of $\mathbf{\scriptsize{W}}$ is either in the same or opposite direction as $\mathbf{m_{2}-m_{1}}$. Hence, $\mathbf{\scriptsize{W}}$ is proportional to $\mathbf{m_{2}-m_{1}}$, showing that the maximization of the class separation criterion leads to the result $\mathbf{\scriptsize{W}}\varpropto \mathbf{m_{2}-m_{1}}$ 

## **Q2:**

The Fisher criterion is 
$$
J(\mathbf{\scriptsize{W}})=\frac{(m_{2}-m_{1})^{2}}{s_{1}^{2}+s_{2}^{2}}
$$
Because $m_{k}=\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{m}_{\mathbf{k}}$, we can get:
$$
(m_{2}-m_{1})^{2}=(\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{m}_{\mathbf{2}}-\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{m}_{\mathbf{1}})^{2}=\mathbf{\scriptsize{W}}^{\mathbf{T}}(\mathbf{m_{2}-m_{1}})(\mathbf{m_{2}-m_{1}})^{\mathbf{T}}\mathbf{\scriptsize{W}}
$$
The between-class matrix is defined as :
$$
\mathbf{S_{B}}=(\mathbf{m_{2}-m_{1}})(\mathbf{m_{2}-m_{1}})^{\mathbf{T}}
$$
So
$$
(m_{2}-m_{1})^{2}=\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{S_{B}}\mathbf{\scriptsize{W}}
$$
Because $m_{k}=\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{m}_{\mathbf{k}}$, $s_{k}^{2}=\sum_{n\in C_{k}}(y_{n}-m_{k})^{2}$ and $y=\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{x}$, we can get:
$$
s_{1}^{2}+s_{2}^{2}=\sum_{n\in C_{1}}(y_{n}-m_{1})^{2}+\sum_{n\in C_{2}}(y_{n}-m_{2})^{2}\\
=\sum_{n\in C_{1}}(\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{x_{1}}-\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{m}_{\mathbf{1}})^{2}+\sum_{n\in C_{2}}(\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{x_{2}}-\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{m}_{\mathbf{2}})^{2}\\
=\mathbf{\scriptsize{W}}^{\mathbf{T}}(\sum_{n\in C_{1}}(\mathbf{x_{1}}-\mathbf{m}_{\mathbf{1}})^{2}+\sum_{n\in C_{2}}(\mathbf{x_{2}}-\mathbf{m}_{\mathbf{2}})^{2})\mathbf{\scriptsize{W}}
$$
The within-class scatter matrix is defined as:
$$
\mathbf{S_{W}}=\sum_{n\in C_{1}}(\mathbf{x_{1}}-\mathbf{m}_{\mathbf{1}})^{2}+\sum_{n\in C_{2}}(\mathbf{x_{2}}-\mathbf{m}_{\mathbf{2}})^{2}
$$
So 
$$
s_{1}^{2}+s_{2}^{2}=\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{S_{W}}\mathbf{\scriptsize{W}}
$$
So, the Fisher criterion can be written in the form:
$$
J(\mathbf{\scriptsize{W}})=\frac{\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{S_{B}}\mathbf{\scriptsize{W}}}{\mathbf{\scriptsize{W}}^{\mathbf{T}}\mathbf{S_{W}}\mathbf{\scriptsize{W}}}
$$

## Q3:

The likelihood function  for the dataset:
$$
L=p(t_n|\phi_n,C_{k})=\prod_{n=1}^{N}\prod_{k=1}^{K}p(C_{k})^{t_{nk}}·p(\phi_n|C_{k})^{t_{nk}}
$$
Take the natural logarithm of the likelihood function, we can get the log-likelihood function:
$$
\ln p(t_n|\phi_n,C_{k})=\sum^{N}_{n=1}\sum^{K}_{k=1}t_{nk}[\ln \pi_{k}+\ln p(\phi_n|C_{k})]
$$
In order to maximize the log-likelihood, we need to preserve the fact that $\sum^{K}_{k=1}\pi_{k}$=1,  so we introduce a Lagrange multiplier as follows:
$$
L=\sum^{N}_{n=1}\sum^{K}_{k=1}t_{nk}[\ln \pi_{k}+\ln p(\phi_n|C_{k})]-\lambda(\sum^{K}_{k=1}\pi_{k}-1)
$$
To obtain the maximum likelihood, differentiate the log-likelihood with respect to $\pi_{k}$ and set the derivative to $0$ :
$$
\sum^{N}_{n=1}\frac{t_{nk}}{\pi_{k}}-\lambda=0
$$
 Therefore,
$$
\sum^{N}_{n=1}t_{nk}=\pi_{k}N+\lambda
$$

$$
\pi_{k}=\frac{N_{k}}{N}
$$

Therefore, the maximum-likelihood solution for the prior probabilities is given by $\pi_{k}=\frac{N_{k}}{N}$, where $N_{k}$ is the number of data points assigned to class $C_{k}$.

## Q4: 

$$
\frac{d\sigma}{da}=\frac{d(\frac{1}{1+\exp(-a)})}{da}\\=-\frac{1}{(1+\exp(-a))^{2}}(0-\exp(-a))\\=\frac{\exp(-a)}{(1+\exp(-a))^{2}}\\
=\frac{1}{1+\exp(-a)}\frac{\exp(-a)}{1+\exp(-a))}\\=
\frac{1}{1+\exp(-a)}\frac{1+\exp(-a)-1}{1+\exp(-a))}\\
\frac{1}{1+\exp(-a)}(1-\frac{1}{1+\exp(-a))})\\
=\sigma(1-\sigma)
$$

## Q5:

The error function for the logistic regression model is given by
$$
E(\mathbf{\scriptsize{W}})=-\ln p(\mathbf{t}|\mathbf{\scriptsize{W}})=-\sum^{N}_{n=1}[t_N\ln y_n+(1-t_{n})\ln(1-y_{n})]
$$
where $y_{n}$ is the model's output and $t_{n}$ is the target. The output can be written as:
$$
y_{n}=\sigma(\mathbf{\scriptsize{W}}^{\mathbf{T}}\phi_{n})
$$
where $\sigma$ is the logistic sigmoid function and $\phi_{n}$ is the $n_{th}$ input vector.

Taking the derivative of the error function with respect to the weights $\mathbf{\scriptsize{W}}$, we can get:
$$
\frac{\partial E(\mathbf{\scriptsize{W}})}{\partial \mathbf{\scriptsize{W}}}=\frac{\partial E(\mathbf{\scriptsize{W}})}{\partial y_{n}}\frac{\partial y_{n}}{\partial (\mathbf{\scriptsize{W}}^{\mathbf{T}}\phi_{n})}\frac{\partial (\mathbf{\scriptsize{W}}^{\mathbf{T}}\phi_{n})}{\partial \mathbf{\scriptsize{W}}}
$$

$$
\frac{\partial E(\mathbf{\scriptsize{W}})}{\partial y_{n}}=-\sum^{N}_{n=1}(\frac{t_{n}}{y_{n}}-\frac{1-t_{n}}{1-y_{n}})
$$

Using the result $\frac{d\sigma}{da}=\sigma(1-\sigma)$ for the derivative of logistic sigmoid,
$$
\frac{\partial y_{n}}{\partial (\mathbf{\scriptsize{W}}^{\mathbf{T}}\phi_{n})}=y_{n}(1-y_{n})
$$
And the last component is :
$$
\frac{\partial (\mathbf{\scriptsize{W}}^{\mathbf{T}}\phi_{n})}{\partial \mathbf{\scriptsize{W}}}=\phi_{n}
$$
So
$$
\frac{\partial E(\mathbf{\scriptsize{W}})}{\partial \mathbf{\scriptsize{W}}}=-\sum^{N}_{n=1}(\frac{t_{n}}{y_{n}}-\frac{1-t_{n}}{1-y_{n}})y_{n}(1-y_{n})\phi_{n}\\=\sum^{N}_{n=1}(y_{n}-t_{n})\phi_{n}
$$
In conclusion, the derivative of the error function for the logistic regression model is given by
$$
\nabla E(\mathbf{\scriptsize{W}})=\sum^{N}_{n=1}(y_{n}-t_{n})\phi_{n}
$$

## Q6:

1. Using $(c-1)$ Linear Discriminant Functions

   Consider three classes $C_{1}$, $C_{2}$, $C_{3}$ in two dimensions. Using two discriminant functions $y_{1}(x),y_{2}(x)$:

   - $y_{1}(x)>0$ for $C_{1}$, $y_{1}(x)<0$ for not in $C_{1}$
   - $y_{2}(x)>0$ for $C_{2}$, $y_{2}(x)<0$ for not in $C_{2}$

   Ambiguity:

   In regions where $y_{1}(x)<0$ and $y_{2}(x)<0$, the classification is ambiguous, as it's unclear whether it belongs to class $C_{1}$, $C_{2}$, or $C_{3}$.

2. Using $c(c-1)/2$ Discriminant Functions

   Consider three classes $C_{1}$, $C_{2}$, $C_{3}$ in two dimensions. Using two discriminant functions $y_{12}(x),y_{13}(x),y_{23}(x)$:

   - $y_{12}(x)>0$ for $C_{1}$, $y_{12}(x)<0$ for $C_{2}$
   - $y_{13}(x)>0$ for $C_{1}$, $y_{13}(x)<0$ for $C_{3}$
   - $y_{23}(x)>0$ for $C_{2}$, $y_{23}(x)<0$ for $C_{3}$

   Ambiguity:

   In regions where  $y_{12}(x)<0$, $y_{13}(x)<0$ and $y_{23}(x)<0$, the classification is ambiguous, leading to uncertainty in assigning the input to one of the three classes.

## Q7:

1.Suppose the convex hull of two sets of points, {$\mathbf{x}^{n}$} and {$\mathbf{z}^{m}$}, intersect. This means that there exists some points {$\mathbf{p}$} can be written as a convex combination of both sets of points:


$$
p=\sum_{n}\alpha^{n}\mathbf{x}^{n}=\sum_{m}\beta^{m}\mathbf{z}^{m}
$$
where
$$
\alpha^{n}>=0,\sum_{n}\alpha^{n}=1
$$

$$
\beta^{m}>=0,\sum_{m}\beta^{m}=1
$$

Assume that the two sets are linearly separable, that means there exists a vector $\hat {\mathbf{\scriptsize{W}}}$ and a scalar $w_{0}$ such that $\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{x}^{n}+w_{0}>0$ for all $\mathbf{x}^{n}$ , and $\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{z}^{m}+w_{0}<0$ for all $\mathbf{z}^{m}$. Then, for points {$\mathbf{p}$}, 
$$
\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{p}+w_{0}=\hat {\mathbf{\scriptsize{W}}}^{T}(\sum_{n}\alpha^{n}\mathbf{x}^{n})+w_{0}=\sum_{n}\alpha^{n}\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{x}^{n}+w_{0}>0
$$
and
$$
\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{p}+w_{0}=\hat {\mathbf{\scriptsize{W}}}^{T}(\sum_{m}\beta^{m}\mathbf{z}^{m})+w_{0}=\sum_{n}\beta^{m}\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{z}^{m}+w_{0}<0
$$
But this is a contradiction, and thus the two sets of points can't be linearly separable.

2.Conversely, suppose the convex hull of two sets of points, {$\mathbf{x}^{n}$} and {$\mathbf{z}^{m}$}, are linearly separable. Then as discussed, there exists a vector $\hat {\mathbf{\scriptsize{W}}}$ and a scalar $w_{0}$ such that $\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{x}^{n}+w_{0}>0$ for all $\mathbf{x}^{n}$ , and $\hat {\mathbf{\scriptsize{W}}}^{T}\mathbf{z}^{m}+w_{0}<0$ for all $\mathbf{z}^{m}$.  If the convex hull intersect, there exists some points, {$\mathbf{p}$} in the intersection such that {$\mathbf{p}$} can be written as a convex combination of both sets. But as the above derived, this leads to a contradiction, thus the convex hulls can't be intersect.

In conclusion,  if their convex hulls intersect, the two sets of points cannot be linearly separable, and conversely that, if they are linearly separable, their convex hulls do not intersect.