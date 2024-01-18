<h1 align = "center">HW3</h1>

<center>12011923 张旭东</center>

### Q1

Define 
$$
R=diag(r_{1},r_{2},...,r_{N})
$$
Then
$$
E_{D}(\mathbf{\scriptsize{W}})=\frac{1}{2}(t-\varnothing \mathbf{\scriptsize{W}})^{\mathbf{\scriptsize{T}}}R(t-\varnothing \mathbf{\scriptsize{W}})
$$
Let
$$
\frac{dE_{D}(\mathbf{\scriptsize{W}})}{d\mathbf{\scriptsize{W}}}=0
$$
We can get
$$
\varnothing^{\mathbf{\scriptsize{T}}}R\varnothing \mathbf{\scriptsize{W}}+\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\varnothing^{\mathbf{\scriptsize{T}}}R\varnothing=t^{\mathbf{\scriptsize{T}}}R\varnothing+\varnothing^{\mathbf{\scriptsize{T}}}Rt
$$
Then
$$
\mathbf{\scriptsize{W}}^{*}=(\varnothing^{\mathbf{\scriptsize{T}}}R\varnothing)^{-1}\varnothing^{\mathbf{\scriptsize{T}}}Rt
$$
**(1)** According to the formula **(5)**, we can get
$$
\beta(t|\mathbf{\scriptsize{X}},\mathbf{\scriptsize{W}},\beta)=\prod^{N}_{n=1}N(t_{n}|\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\varnothing(\mathbf{\scriptsize{X_{n}}}),\beta^{-1})
$$

$$
\ln p(t|\mathbf{\scriptsize{W}},\beta)=\frac{N}{2}\ln\beta-\frac{N}{2}\ln(2\pi)-\beta E_{D}^{’}(\mathbf{\scriptsize{W}})
$$

$$
E_{D}^{’}(\mathbf{\scriptsize{W}})=\frac{1}{2}\sum^{N}_{n=1}(t_{n}-\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\varnothing(\mathbf{\scriptsize{X_{n}}}))^{2}
$$

Therefore, $r_{n}$ can be regarded as $\beta$ parameter particular to the data point $(\mathbf{\scriptsize{X_{n}}},t_{n})$  

**(2)** $r_{n}$ can be regarded as an effective number of replicated observations of data point $(\mathbf{\scriptsize{X_{n}}},t_{n})$. If $r_{n}$ is positive integer, it's more clear to be interpreted this way.

### Q2

$$
\begin{align}
\ln p(\mathbf{\scriptsize{W}},\beta|t)=\ln p(\mathbf{\scriptsize{W}},\beta)+\sum^{N}_{n=1}\ln p(t_{n}|\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\varnothing(\mathbf{\scriptsize{X_{n}}}),\beta^{-1})\\
=\frac{M}{2}\ln \beta-\frac{1}{2}\ln\mathbf{S_{0}}-\frac{\beta}{2}(\mathbf{\scriptsize{W}}-\mathbf{m_{0}})^{\mathbf{\scriptsize{T}}}\mathbf{S_{0}}^{-1}(\mathbf{\scriptsize{W}}-\mathbf{m_{0}})-b_{0}\beta\\
+(a_{0}-1)\ln\beta+\frac{N}{2}\ln\beta-\frac{\beta}{2}\sum^{N}_{n=1}(\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}\varnothing(\mathbf{\scriptsize{X_{n}}})-t_{n})^{2}+C
\end{align}
$$

where $C$ is a constant.
$$
p(\mathbf{\scriptsize{W}},\beta|\mathbf{t})=p(\mathbf{\scriptsize{W}}|\beta,\mathbf{t})p(\beta|\mathbf{t})
$$
consider the first dependence on $\mathbf{\scriptsize{W}}$, we have
$$
\ln p(\mathbf{\scriptsize{W}}|\beta,\mathbf{t})=-\frac{\beta}{2}\mathbf{\scriptsize{W}}^{\mathbf{\scriptsize{T}}}[\varnothing^{\mathbf{\scriptsize{T}}}\varnothing+\mathbf{S_{0}^{-1}}]\mathbf{\scriptsize{W}}+\mathbf{\scriptsize{W}}^\mathbf{\scriptsize{T}}[\beta\mathbf{S_{0}^{-1}}\mathbf{m_{0}}+\beta\varnothing^\mathbf{\scriptsize{T}}t]+C
$$
Therefore, $p(\mathbf{\scriptsize{W}}|\beta,t)$ is a Gaussian distribution with mean and covariance given by 
$$
\mathbf{m_{N}}=\mathbf{S_{N}}[\mathbf{S_{0}}^{-1}\mathbf{m_{0}}+\varnothing^{\mathbf{\scriptsize{T}}}t]
$$

$$
\beta\mathbf{S_{N}}^{-1}=\beta(\mathbf{S_{0}}^{-1}+\varnothing^{\mathbf{\scriptsize{T}}}\varnothing)
$$

$$
\mathbf{S_{N}}=(\mathbf{S_{0}}^{-1}+\varnothing^{\mathbf{\scriptsize{T}}}\varnothing)^{-1}
$$

$$
\ln p(\beta|t)=-\frac{\beta}{2}\mathbf{m_{0}}^{\mathbf{\scriptsize{T}}}\mathbf{S_{0}}^{-1}\mathbf{m_{0}}+\frac{\beta}{2}\mathbf{m_{N}}^{\mathbf{\scriptsize{T}}}\mathbf{S_{N}}^{\mathbf{\scriptsize{T}}}\mathbf{m_{N}}+\frac{N}{2}\ln\beta-b_{0}\beta+(a_{0}-1)\ln\beta-\frac{\beta}{2}\sum^{N}_{n=1}t_{n}^{2} +C
$$

We recognize the $\log$ of Gamma distribution
$$
\Gamma(\beta|a_{N},b_{N})=\frac{b_{N}^{a_{N}} \beta^{a_{N}-1}e^{-b_{N}\beta}}{\Gamma(a_{N})}
$$

$$
\log (\Gamma(\beta|a_{N},b_{N}))=a_{N}\log b_{N}+(a_{N}-1)\log\beta-b_{N}\beta-log(\Gamma(a_{N}))
$$

$$
a_{N}=a_{0}+\frac{N}{2}
$$

$$
b_{N}=b_{0}+\frac{1}{2}(\mathbf{m_{0}}^{\mathbf{\scriptsize{T}}}\mathbf{S_{0}}^{-1}\mathbf{m_{0}}+\mathbf{m_{N}}^{\mathbf{\scriptsize{T}}}\mathbf{S_{N}}^{-1}\mathbf{m_{N}}+\sum^{N}_{n=1}t_{n}^{2})
$$

**Q3**
$$
\int \exp{[E(\mathbf{\scriptsize{W}})]}d{\mathbf{\scriptsize{W}}}=\int \exp[-E(\mathbf{m_{N}})-\frac{1}{2}(\mathbf{\scriptsize{W}}-\mathbf{m_{N}})^{\mathbf{\scriptsize{T}}}\mathbf{A}(\mathbf{\scriptsize{W}}-\mathbf{m_{N}})]d{\mathbf{\scriptsize{W}}}\\
=\exp[-E(\mathbf{m_{N}})]\int\exp[-\frac{1}{2}(\mathbf{\scriptsize{W}}-\mathbf{m_{N}})^{\mathbf{\scriptsize{T}}}\mathbf{A}(\mathbf{\scriptsize{W}}-\mathbf{m_{N}})]d{\mathbf{\scriptsize{W}}}\\
=\exp[-E(\mathbf{m_{N}})](2\pi)^{\frac{\mathbf{m}}{2}}|\mathbf{A}|^{-\frac{1}{2}}\int\frac{1}{(2\pi)^{\frac{\mathbf{m}}{2}}}\frac{1}{|\mathbf{A}|^{-\frac{1}{2}}}\exp[{-\frac{1}{2}}(\mathbf{\scriptsize{W}}-\mathbf{m_{N}})^{\mathbf{\scriptsize{T}}}\mathbf{A}(\mathbf{\scriptsize{W}}-\mathbf{m_{N}})]d{\mathbf{\scriptsize{W}}}\\
=\exp[-E(\mathbf{m_{N}})](2\pi)^{\frac{\mathbf{m}}{2}}|\mathbf{A}|^{-\frac{1}{2}}
$$

$$
p(\mathbf{t}|\alpha,\beta)=(\frac{\beta}{2\pi})^{\frac{N}{2}}(\frac{\alpha}{2\pi})^{\frac{M}{2}}\int\exp[E(\mathbf{\scriptsize{W}})]d{\mathbf{\scriptsize{W}}}\\
=(\frac{\beta}{2\pi})^{\frac{N}{2}}(\frac{\alpha}{2\pi})^{\frac{M}{2}}\exp[E(\mathbf{m_{N}})](2\pi)^{\frac{M}{2}}|\mathbf{A}|^{-\frac{1}{2}}
$$

$$
\ln p(\mathbf{t}|\alpha,\beta)=\frac{M}{2}\ln\alpha+\frac{N}{2}\ln\beta-E(\mathbf{m_{N}})-\frac{1}{2}\ln|\mathbf{A}|-\frac{N}{2}\ln(2\pi)
$$

**Q4**
$$
\frac{\partial F(a)}{\partial a}=\sum_{i}(Y_{i}-aX_{i})(-X_{i})=\sum_{i}aX_{i}-X_{i}Y_{i}
$$

$$
a=\frac{\sum_{i}X_{i}Y_{i}}{\sum_{i}X_{i}^{2}}
$$

**Q5**
$$
\log p(y|\theta)=y\log\theta-\theta-\sum^{y}_{i=0}\log i\\
=\sum^{n}_{i=1}(y_{i}\log\theta-\theta-\log y_{i}!)\\
=\sum_{i=1}^{n}(y_{i}\log\theta-\log y_{i}!)-n\theta
$$
**Q6**
$$
\log f_{X}(x)=\alpha \log \lambda+(\alpha-1)\log(x)+\lambda x-\log\Gamma(\alpha)
$$

$$
g(\lambda)=n\alpha \log(\lambda)+(\alpha-1)\log(\prod^{n}_{i=1}X_{i})-\lambda\sum^{n}_{i=1}X_{i}-n\log\Gamma(\alpha)
$$

$$
\frac{d g(\lambda)}{d\lambda}=\frac{n\alpha}{\lambda}-\sum^{n}_{i=1}X_{i}
$$

$$
\hat \lambda=\frac{\alpha}{\frac{1}{n}\sum^{n}_{i=1}X_{i}}
$$

