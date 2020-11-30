---
layout: post
title:  'Variational inference'
date: 08-07-2019
img_link: /assets/3_variational_approx.png
excerpt: 'In the last post, we defined generative models and we learned how to build them, however, one important piece was missing: How do we work with them? How do we infer the parameters or the variables of interest of the model? I mentioned briefly that there are many ways to do inference on these variables. In this post I try to explain how one of this inference methods work.'
---
## Recap from the last post
Remember, from the last post that we create a generate model based on observed and latent variables and we would like to infer the distribution of these latent variables using the Bayes theorem. Let $$x$$ be the set of observed variables and $$z$$ the set of variables we want to do inference on:

$$
p(z \lvert x) = \frac{p(z, x)}{p(x)} = \frac{p(x \lvert z)p(z)}{p(x)} = \frac{p(x \lvert z)p(z)}{\int p(z, x)dz}\qquad \text{Bayes theorem}
$$

Let's study this equation in a bit more detail with a couple of examples. The term $$p(x \lvert z)$$ is called the **likelihood** function, the likelihood function is defined by the modeler and expresses the probability of observing $$x$$ under the distribution arising from the latent variable $$z$$. For example, the modeler thinks that a each observed variable from a sample of size $$N$$, $$x_{n}$$, is distributed as a normal distribution with known mean $$\mu=0$$ and unknown variance $$z_{n}$$ Then, assuming independently and identically distributed (i.i.d.) data, the likelihood function can be expressed as: $$p(x \lvert z)=\prod_{n}^{N}\mathcal{N}(x_{n}\lvert \mu=0, \, z_{n})$$. Alternatively, I could have chosen the distribution to have a *global* latent variable $$z$$ instead of a *local* one $$z_{i}$$, or I could have chosen the distribution to be a uniform one with minimum bound $$0$$ and maximum bound $$z$$. Note also that, in general, the likelihood is easy to evaluate given a value of $$z$$.

The second term $$p(z)$$  is called the **prior**. It is also a choice of the modeler and it contains the modeler's *prior* belief of how should a latent variable be distributed. Priors can be informative (bearing a lot of information of $$z$$), or non-informative. Let's assume, for our model, that the $$z$$ is distributed as an exponential random variable with rate parameter ($$\lambda$$) $$5$$. Our prior probability (keeping the assumption of a local latent variable) would look like this: $$p(z)=\prod_{n}^{N}Exp(z_{n}\lvert\lambda=5)$$. Just as in the likelihood case, we could have assumed other distributions or other parameters.

Finally, the denominator of the Bayes theorem, the **evidence** is the distribution of $$x$$, which can be acquired by integrating over the product of the terms we learned on the previous two paragraphs: $$p(x) = \int p(z, x)dz = \int \prod_{n}^{N}\mathcal{N}(x_{n}\lvert \mu , \, z_{n})Exp(z_{n}\lvert\lambda=5)dz$$. This integral, however, is generally intractable and therefore we have to appeal to approximate inference methods. This is the main motivation to do variational inference in a probabilistic model.

## What is variational inference?

We just showed why we care about developing approximate inference. Now, what exactly is approximate inference? approximate inference (as you might have guessed), is a way to approximate the posterior distribution of our latent variables. There are several ways to do so, however (and arguably) the most popular ways to do so are using Variational Inference (VI) and Markov Chain Monte Carlo (MCMC). This post focuses solely on VI. We however, will learn what Monte Carlo methods are, because they are needed for efficient VI.

The main idea of variational inference is to turn the __*inference*__ problem into an __*optimization*__ problem. The way to do this is by introducing an approximating distribution $$q_{\phi}(z)$$ from a family of distributions which is easy to optimize. By family of distributions I mean any distribution that is parametrized by a set of parameters $$\Theta$$. For example, the normal distribution is a family of distributions parametrized by a mean and a variance. These parameters can be easily optimized for a sample of observations. The point of VI is to find the **variational**  parameters $$\phi$$ that best approximate the posterior distribution.

<div class="blogViewImg">
  <figure>
    <img src="{{ site.baseurl }}/assets/3_optimization.png" alt="" />
    <figcaption markdown="span">The variational inference problem. We want to approximate $$p(z|x)$$ using a distribution from the family $$q(z)$$ parametrized by $$\nu$$ (in this post we use $$\phi$$). The optimal distribution is picked (in this case) using the Kullback-Leibler divergence. Source: <a href="https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf" target="_blank">(Blei et.al. 2016)</a></figcaption>
  </figure>
</div>

For me, this was a difficult concept to grasp at the beginning: how are we going to optimize an approximate distribution to our goal distribution if we don't even know our posterior distribution? The point is that you can do so in **expectation**, and therefore, you can approximate these expectations using Monte Carlo methods (don't worry, I will explain that later). There are two proofs I know that arrive to the same conclusion. The first one that starts from the Kullback-Leibler [divergence measure](https://en.wikipedia.org/wiki/Divergence_(statistics)), a non-symmetric measure of how different two distributions are. And the second which starts from the log of the data evidence. Let's begin with the Kullback-Leibler divergence proof:

$$
 KL(q_{\phi}(z)||p(z|x)) = \int q_{\phi}(z) \log \frac{q_{\phi}(z)}{p(z|x)}\qquad \text{Definition of the KL divergence} \\
 = \mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)] - \mathbb{E}_{q_{\phi}(z)}[\log p(z|x)] \\
 = \mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)] - \mathbb{E}_{q_{\phi}(z)}\bigg[\log \frac{p(x, z)}{p(x)}\bigg]\qquad \text{Bayes theorem} \\
 = \mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)] - \mathbb{E}_{q_{\phi}(z)}[\log p(x, z)] + \mathbb{E}_{q_{\phi}(z)}[\log p(x)] \\
$$

We would like to *minimize* the KL-divergence with respect to $$\phi$$. This will arrive to the best variational distribution $$q_{\phi}(z)$$ given the chosen variational family. The lower value that the KL-divergence can take is 0, that happens when $$q_{\phi}(z) = p(z\lvert x)$$. This is, when we have chosen a variational family that effectively contains the posterior distribution and we were able to optimize it perfectly. Note that, since the last term on the right hand side of the equation does not depend on $$\phi$$, it is not necessary in the optimization procedure. After the second proof (which now follows), I will explain what do the terms on the last equation mean. For the second proof, we can start from the log-evidence of the data. This is a quantity we would like to *maximize*:

$$
 \log p(x) = \log \int p(x, z)dz \\
= \log \int q_{\phi}(z) \frac{p(x, z)}{q_{\phi}(z)}dz \\
= \log \mathbb{E}_{q_{\phi}(z)}\bigg[\frac{p(x, z)}{q_{\phi}(z)}\bigg] \\
\geq \mathbb{E}_{q_{\phi}(z)}\bigg[\log \frac{ p(x, z)}{q_{\phi}(z)}\bigg] \qquad \text{by Jensen's inequality} \\
= \mathbb{E}_{q_{\phi}(z)}[\log p(x, z)]-\mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)]
$$

The last equation is known as the Variational lower bound, the negative free energy $$-\mathcal{F}$$ or the Evidence Lower BOund (ELBO, $$\mathcal{L}$$). As mentioned before, we want to maximize this (a lower bound on the evidence) to find the parameters of the distribution $$q_{\phi}(z)$$ that approximate the likelihood of our data best. The reason of why this is called Variational lower bound or Evidence Lower BOund is because, as mentioned before, the KL-divergence is always positive and thus always adds to the evidence of the data $$p(x)$$. Let's briefly analyze what the terms in the ELBO mean. The first term, the expected complete log-likelihood prefers that $$q_{\phi}$$ is on the Maximum A Posteriori (MAP) solution, meaning, placing the mass centered at one point while the second term, the negative entropy, encourages $$q_{\phi}$$ to be diffuse or, in other words, as close as possible to the uniform distribution.

<div class="blogViewImg">
  <figure>
    <img src="{{ site.baseurl }}/assets/3_variational_approx.png" alt="" />
    <figcaption>Graphical depiction of variational inference for a 1-dimensional case. A complex distribution of two modes is approximated with a single distribution.</figcaption>
  </figure>
</div>

We can see VI as a two step process. First we need to define an approximating distribution, $$q_{\phi}(z)$$, and second, we need to find an appropriate optimization scheme to find the parameters $$\phi$$ that gives us the best ELBO.

About the choice of the variational (approximating) distribution. We would like to choose a distribution easier to work with than the distribution we are trying to approximate (otherwise we would just do inference on the original one, right?). There are many ways to define the variational distribution and it is ultimately the choice of the researcher. Note that when picking a variational family, we can represent the distribution in a graphical model just as in the previous post. In figure 3 we draw the graphical representation of an arbitrary variational distribution. There is a subtlety about choosing the distribution, though. It is necessary to define *all* the latent variables in the variational distribution, you will notice this whenever you are programming this in probabilistic programs that allow to define a variational distribution for your model (Edward, Pyro, etc.).

Now, with respect to the optimization procedure, we know from optimization theory that the parameters that optimize a function are found when the gradient of the function with respect to the parameter we want to optimize equals zero. In our case:

$$
\nabla_{\phi}[\mathbb{E}_{q_{\phi}(z)}[\log p(x, z)]-\mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)]] \\
=\nabla_{\phi}\mathbb{E}_{q_{\phi}(z)}[\log p(x, z)]-\nabla_{\phi}\mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)]=0
$$

During the first years of VI, the steps a researcher had to follow in order to do inference in their models were the following:
1. Setting a variational distribution $$p_{\phi}(z)$$ as a function of $$\phi$$. One natural way to set the approximating distribution is setting a distribution for each latent variable independent of all the rest of them. This is called the mean-field distribution.
2. Computing $$\mathbb{E}_{q_{\phi}(z)}[\log p(x, z)]$$ and $$\mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)]$$, that is, writing explicitly the distribution of your model, taking a constant expected value of anything that is not defined by $$q_{\phi}(z)$$ and use $$q_{\phi}(z)$$ for all $$z$$. This step allowed researchers to build the ELBO.
3. Take the gradient of the ELBO with respect to each $$\phi$$ and set it to zero to find the parameter updates they were going to use in their optimization algorithm.
4. Use coordinate gradient ascent to optimize $$\phi$$.

<div class="blogViewImg">
  <figure>
    <img src="{{ site.baseurl }}/assets/3_mean_field.png" alt="" />
    <figcaption markdown="span">Example of a graph of a latent variable model (left) and graph of the inference model (right). The inference model is parametrized as a mean field variational family. This is, a distribution where each latent variable is independent of each other. The variational parameters are represented as white squares. Ideally, we would like the distribution on the right to approximate the distribution on the left under some divergence measure. A commonly used divergence is the Kullback-Leibler divergence. Adapted from: <a href="http://www.cs.columbia.edu/~blei/papers/Blei2014b.pdf" target="_blank">(Blei 2014)</a></figcaption>
  </figure>
</div>

This process, even though mathematically elegant, was error prone and potentially prohibitevely expensive to compute in cases where there are local variables, global variables and several observed points. On top of that, this procedure imposed some restrictive conditions on the variational family that could be used in order to follow the steps. VI research put a strong focus for many years on how to make this process faster and less error prone. Before explaining this advances on this point I have to make a short explanation of something else.

## A short digression into Monte Carlo integration
Before going any further into the explanation of how to do VI in graphical models I would like to introduce (what I believe) one of the most powerful methods in mathematics: Monte Carlo integration. This is a particular application of a wider family of methods that are simply the "Monte Carlo methods". The idea is to approximate a hard (or intractable) integral using random numbers (coming from a distribution). Suppose we have a distribution $$p(x)$$ of a random variable $$x$$. Suppose also we have a function of this random variable $$f(x)$$. With Monte Carlo integration we are able to compute $$\int p(x) f(x) dx$$ as long as we can make draws from $$p(x)$$. Note that, since $$p(x)$$ is a distribution, we can rewrite this integral as:  $$\int p(x) f(x) dx=\mathbb{E}_{p(x)}[f(x)]=\mathbb{E}_{x\sim p(x)}[f(x)]$$ (See the similarities between this and the way we derived the ELBO?). The steps to do the approximation are the following:
1. Make N draws $$x^{i} \sim p(x)$$ where $$i=1 \ldots N$$
2. Transform $$x^{i}$$ with $$f(x)$$
3. Average the results of the transformations $$\frac{1}{N} \sum_{i}^{N}f(x^{i})$$

As $$N$$ approaches to infinity the approximation is better. One particular case of this procedure is the computation of the mean of the distribution $$p(x)$$. In that case we only need to set $$f(x)=x$$ and that's it!

## Super fast VI with stochastic optimization and black-box VI
The main idea of stochastic optimization is to take noisy (Monte Carlo) estimates of the gradient. Why we would like to do that? because we could use gradient based optimization procedures. And how can we do that? we need to "push" the gradient with respect to the variational parameters $$\nabla_\phi$$ inside the expectation of the log-likelihood $$\mathbb{E}_{q_{\phi}(x)}[\log p(x, z)]$$ and the entropy of the variational distribution $$\mathbb{E}_{q_{\phi}(x)}[\log q(z)]$$. There are different ways this has been explored in the literature, however, on this blog post I am going to focus on black-box variational inference as in (Ranganath et.al., 2013). Something important to take into account: This method has been proposed in other areas with several different names which might be useful to know such as the score function estimator, the likelihood ratio estimator or the REINFORCE estimator.

Let's proof how we can "push" the derivative inside the expectation under mild assumptions. First, let's begin with the derivative of the definition of the ELBO in

$$
\nabla_{\phi} \mathcal{L} = \nabla_{\phi} [\mathbb{E}_{q_{\phi}(z)}[\log p(x, z)]-\mathbb{E}_{q_{\phi}(z)}[\log q_{\phi}(z)]] \\
=\nabla_{\phi} \int q_{\phi}(z)[\log p(x, z)-\log q_{\phi}(z)]dz \\
=\int \nabla_{\phi}  [q_{\phi}(z)[\log p(x, z)-\log q_{\phi}(z)]]dz \\
= \int \nabla_{\phi} q_{\phi}(z)[\log p(x, z)-\log q_{\phi}(z)]dz  \\
+\int q_{\phi}(z)\nabla_{\phi}\log q_{\phi}(z)dz
$$

Before we arrive to our final expression, I want to explain each of the steps in the previous derivation. From the first equation to the second equation is the definition of the expected value. We did this already for the derivation of the ELBO. From the second equation to the third equation (changing the order of the derivative and the integral) can be done by assuming that the derivative of the log-likelihood with respect to $$\phi$$ exists and that both the log-likelihood and that the derivative are bounded. The first condition (and half of the second) is met because $$\log p(x, z)$$ does not depend on $$\phi$$ and so the derivative is zero.  Now the boundedness of the log-likelihood is also met as long as the probability of every point is strictly positive. From the third equation to the fourth equation we used the product rule of differentiation and the sum rule for the second term. We later used the previously mentioned fact that $$\nabla_{\phi}\log p(x, z)=0$$.

The next step is to prove that the second term is zero. To prove this, we are going to use a common used trick in statistics and machine learning called the log-derivative trick (Keep these tricks in mind, you might find them useful in your own research). The log-derivative trick is a use of the chain rule of differentiation. The derivative of the logarithm of a function is: $$\nabla_{\phi}\log q_{\phi}(z)=\frac{\nabla_{\phi}q_{\phi}(z)}{q_{\phi}(z)}$$ and so starting from the derivative of the function we get: $$q_{\phi}(z)\nabla_{\phi}\log q_{\phi}(z)=\nabla_{\phi}q_{\phi}(z)$$. With this in mind, let's take the second term in the sum from our derivation above:

$$
\int q_{\phi}(z)\nabla_\phi\log q_{\phi}(z)dz \\
=\int\nabla_{\phi} q_{\phi}(z)dz \\
=\nabla_{\phi}\int q_{\phi}(z)dz =\nabla_{\phi}1=0
$$

From the third to the fourth equation, we use a similar logic to that on the derivation above, and we use the fact that the integral of a valid probability distribution must integrate to 1. Let's now take care of the first term in the sum:

$$
\int \nabla_{\phi} q_{\phi}(z)[\log p(x, z)-\log q_{\phi}(z)]dz  \\
=\int q_{\phi}(z)\nabla_{\phi}\log q_{\phi}(z)[\log p(x, z)-\log q_{\phi}(z)]dz \\
=\mathbb{E}_{q_{\phi}(z)}[\nabla_{\phi}\log q_{\phi}(z)[\log p(x, z)-\log q_{\phi}(z)]]
$$

Finally we have arrived to our desired expression. We have finally pushed the derivative inside the expectation and we can use Monte Carlo integration in order to approximate the gradient of the expectation with respect to the variational parameters. Notice two things from the previous result. First, in order to use this type of variational inference you need to be able to compute the log-likelihood $$\log p(x, z)$$ and be able to compute the derivative of your chosen variational distribution with respect to the variational parameters. These are relatively mild conditions! The second thing to notice from this approach is that the variance of the estimate of the gradient is high so you have to appeal to some techniques which we don't cover in this post to reduce is such as Rao-Blackwelliaztion and Control Variates. Third, this procedure still does not solve the problem of hierarchical models (models with global and local variables like on the first post) but it is easy to extend it to do so. In this case, you only need to: 1) take a sub set of the observations, 2) compute the noisy gradients of the expectation with respect to the local variable 3) update the local variables using their gradients and 4) update the global variable by scaling up the local variables by the number of observations they are representing (Hoffman et.al., 2012). If you are interested in this, go to the original paper, after understanding this post, it should not be such a hard read.

<div class="blogViewImg">
  <figure>
    <img src="{{ site.baseurl }}/assets/3_speed_gains.png" alt="" />
    <figcaption>Comparison of speed gains between BBVI and a Gibbs sampler approach. Source: <a href="http://proceedings.mlr.press/v33/ranganath14.pdf" target="_blank">(Ranganath et.al. 2013)</a></figcaption>
  </figure>
</div>

The algorithm that summarizes the black-box variational inference procedure is the following:

{% highlight python %}
Input: data x, likelihood function p(x,z), variational distribution q(z), a learn rate alpha
Initialize: phi randomly
While not converged:
	For i in Number of Samples (NS):
		z[i] ~ q(z) (using the current variational parameters)
	Compute the gradient of the ELBO using the formula above and the NS draws from above.
	phi[t+1] = phi[t] + alpha * gradient_ELBO
{% endhighlight %}

## The reparametrization trick: an alternative to BBVI
A second approach I want to explain in this post is that of the reparametrization trick as introduced in (Kingma and Welling, 2014). This method is also called the pathwise estimator. The goal of this method is, again, to push the derivative inside the expectation in order to estimate the gradient of the expectation using Monte Carlo integration. This can be done by representing a random variable $$z$$ as a deterministic variable with a deterministic function $$g$$ and an **auxiliary** random variable $$\epsilon$$ like this: $$z=g_{\phi}(\epsilon)$$. Before going to the main result I would like to remember the [Law of the unconscious statistician (LOTUS)](https://en.wikipedia.org/wiki/Law_of_the_unconscious_statistician). This theorem states that given a function $$g(\epsilon)$$ of a random variable $$\epsilon \sim p(\epsilon)$$ we can compute the expectation of $$g$$ as  $$\mathbb{E}_{\epsilon\sim p(\epsilon)}[g(\epsilon)]=\int g(\epsilon)p(\epsilon)d\epsilon$$. This means that we only need to know the distribution of the random variable $$p(\epsilon)$$ and the function $$g$$ in order to compute expectations of $$g$$. As we have done before, if we can make draws from $$p(\epsilon)$$ we can compute these expectations using Monte Carlo integration.

How is this related to pushing the gradient inside the integral? From the previous sections, we have seen that computing $$\nabla_{\phi}\mathbb{E}_{q_{\phi}(z)}[\log p(x, z)-\log q_{\phi}(z)]$$ can be problematic. Let's start defining $$f(x,z)=\log p(x, z)-\log q_{\phi}(z)$$ and $$z=g(\epsilon, \phi)$$ where $$g$$ is a known function of a random variable $$\epsilon \sim p(\epsilon)$$ we can easily sample from. Using LOTUS, we arrive to

$$\nabla_{\phi} \mathbb{E}_{z\sim q_{\phi}(z)}[f(x,z)]= \nabla_{\phi}\mathbb{E}_{\epsilon \sim p(\epsilon)}[f(x, g(\epsilon, \phi))]=\mathbb{E}_{\epsilon \sim p(\epsilon)}[\nabla_{\phi}f(x, g(\epsilon, \phi))]
$$

The last equality holds because the expectation does not depend directly on the variational parameters. This reparametrization trick can be easily used in three cases:
1. Case one, there is a tractable inverse Cumulative Distribution Function (CDF). In this case, use $$\epsilon \sim \mathcal{U}(0,1)$$, and choose $$g_{\phi}(.)$$ be the inverse CDF of $$q_{\phi}(z)$$. Some examples are: Exponential, Cauchy, Logistic, Rayleigh, Pareto, Weibull, Reciprocal, Gompertz, Gumbel and Erlang distributions.
2. Case two, any location-scale family. We can choose $$\epsilon$$ to be the standard distribution ($$\text{location}=0$$ and $$\text{scale} =1$$) and pick $$g=\text{location}+\text{scale} * \epsilon$$. Examples: Laplace, Elliptical, Student's t, Logistic, Uniform, Triangular and Gaussian distributions.
3. Case three, compositions. Some random variables are expressed as different transformations of auxiliary variables. For example, the Log-Normal (exponentiation of a normal distribution), Gamma (sum over exponentially distributed variables), Dirichlet (weighted sum of gamma distributed variables), Beta, Chi-Squared and F distributions.

Note that, in order to estimate the gradient of the expected value, we need to know $$\nabla_{z}f(x,z)$$. The way we would learn with this reparametrization is identical to that of the BBVI. This type of estimator has empirically lower variance than that of BBVI. This trick is going to be very useful on the next post where I am going to talk about neural networks and VI.

## Wrap up of fast Variational Inference
This section is just a short summary of what we just learned in the previous sections. This is an adapted version of [Blei's presentation](https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf) in NIPS 2016:

<table>
<colgroup>
<col width="50%" />
<col width="50%" />
</colgroup>
  <tr>
    <th>Score function estimator</th>
    <th>Re-parametrization (Pathwise)</th>
  </tr>
  <tr>
    <td markdown="span">Differentiates the variational approximation $$\nabla_{\phi}q_{\phi}(z)$$</td>
    <td markdown="span">Differentiates $$\nabla_{\phi}[\log p(x, g(\epsilon, \phi))-\log q_{\phi}(g(\epsilon, \phi))]$$</td>
  </tr>
  <tr>
    <td markdown="span">Discrete and continous models</td>
    <td markdown="span">Requires differentiable models (No discrete latent variables, since we need continuity in $$\log p(x,z)$$ with respect to $$z$$ in order to estimate $$\nabla_{z}\log p(x,z)$$)</td>
  </tr>
  <tr>
    <td markdown="span">Works for large class of variational approximations</td>
    <td markdown="span">Requires variational approximation to have the form $$z=g(\epsilon, \phi)$$</td>
  </tr>
  <tr>
    <td markdown="span">Variance can be a problem</td>
    <td markdown="span">Generally better behaved variance</td>
  </tr>
</table>

## Other extensions

In future posts, I will write about two subjects that I find interesting in Variational Inference. First, I want to explore the relation between variational inference and neural networks. Particularly I am going to be looking into the Variational Auto-Encoder and how to make richer variational families for more flexible models. In this matter, we will be looking into Normalizing Flows and Hierarchical Variational Inference.

### References
<ol class="bibliography"><li><span id="bishop_2006_pattern">Bishop, C. M. (2006). <i>Pattern recognition and machine learning</i>. springer. Retrieved from <a href="http://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf" target="_blank">link</a></span></li>
<li><span id="blei_2014_build">Blei, D. M. (2014). Build, compute, critique, repeat: Data analysis with latent variable models. <i>Annual Review of Statistics and Its Application</i>, <i>1</i>, 203–232. Retrieved from <a href="http://www.cs.columbia.edu/~blei/papers/Blei2014b.pdf" target="_blank">link</a></span></li>
<li><span id="blei_2016_variational">Blei, D., Ranganath, R., &amp; Mohamed, S. Variational Inference: Foundations and Modern Methods. Retrieved from <a href="https://media.nips.cc/Conferences/2016/Slides/6199-Slides.pdf" target="_blank">link</a></span></li>
<li><span id="hoffman_svi_2013">Hoffman, M. D., Blei, D. M., Wang, C., &amp; Paisley, J. (2013). Stochastic Variational Inference. <i>Journal of Machine Learning Research</i>, <i>14</i>, 1303–1347. Retrieved from <a href="http://jmlr.org/papers/volume14/hoffman13a/hoffman13a.pdf" target="_blank">link</a></span></li>
<li><span id="kingma_vae_2014">Kingma, D. P., &amp; Welling, M. (2014). Auto-Encoding Variational Bayes. In <i>2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings</i>. Retrieved from http://arxiv.org/abs/1312.6114</span></li>
<li><span id="pereira_mbml_2019">Pereira, F., &amp; Rodrigues, F. (2019). Model Based Machine Learning. Danmarks Tekniske Universitet. Retrieved from <a href="https://github.com/Chechgm/42186-model-based-machine-learning" target="_blank">(Unofficial) repo link</a></span></li>
<li><span id="ranganath_2014_black">Ranganath, R., Gerrish, S., &amp; Blei, D. (2014). Black box variational inference. In <i>Artificial Intelligence and Statistics</i> (pp. 814–822). Retrieved from <a href="http://proceedings.mlr.press/v33/ranganath14.pdf" target="_blank">link</a></span></li>
<li><span id="zhang_2018_advances">Zhang, C., Butepage, J., Kjellstrom, H., &amp; Mandt, S. (2018). Advances in variational inference. <i>IEEE Transactions on Pattern Analysis and Machine Intelligence</i>.Retrieved from <a href="https://arxiv.org/pdf/1711.05597.pdf" target="_blank">link</a></span></li></ol>

#### Thanks
I would like to thank Katarzyna Zukowska, Filipe Rodrigues, Mateo Dulce and Aldo Pareja for reading this post and their insightful comments.
