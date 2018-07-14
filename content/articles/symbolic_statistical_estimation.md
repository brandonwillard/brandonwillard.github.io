---
author:
- 'Brandon T. Willard'
bibliography:
- 'symbolic.bib'
date: '2017-01-18'
figure_dir: '{attach}/articles/figures/'
figure_ext: png
title: A Role for Symbolic Computation in the General Estimation of Statistical Models
---

Introduction
============

In this document we describe how symbolic computation can be used to provide generalizable statistical estimation through a combination of existing open source frameworks. Specifically, we will show how symbolic tools can be used to address the estimation of non-smooth functions that appear in models with parameter regularization, shrinkage and sparsity. We employ a mathematical framework that makes extensive use of *proximal operators* [@parikh_proximal_2014; @combettes_proximal_2011] and their properties for maximum a posteriori (MAP) estimation: i.e. the *proximal framework*. This framework produces what we’ll call *proximal methods* and their implementations as *proximal algorithms*.

In @polson_proximal_2015 we outlined a set of seemingly disparate optimization techniques within the fields of statistics, computer vision, and machine learning (e.g. gradient descent, ADMM, EM, Douglas-Rachford) that are unified by their various applications of proximal methods. These methods–and the concepts behind them–have found much success in recent times and admit quite a few interesting paths for research. In other words, there are many reasons to alone discuss the implementation of proximal methods.

Proximal operators also enjoy a breadth of closed-form solutions and useful properties that are amenable to symbolic computation. In more than a few cases, the work required to produce a proximal algorithm overlaps with well-established features of computer algebra systems and symbolic mathematics, such as symbolic differentiation and algebraic equation solving.

Symbolic integration provides an excellent example of how proximal operators could be implemented in a symbolic system. In these systems, mappings between functions (as canonicalized graphs) and their generalized hypergeometric equivalents are used to exploit the latter’s relevant convolution identities. In the same vein, it is possible to use tables of closed-form proximal operators and their properties to produce a wide array of estimation algorithms for many non-smooth functions. We outline how this might be done in the following sections.

Otherwise, the ideas discussed here are part of a never-ending attempt to answer a question that arises naturally in both mathematics and programming–at all levels: *How does one provide a means of generating robust solutions to as many problems as possible?* Instead of the common efforts to independently implement each model, method and/or combination of the two–followed by their placement in an API or library of functions–implementations can be encoded in and organized by the very mathematics from which they were derived. This close coupling between mathematical principles and their implementations might be the only reasonable way to remove barriers between theory, research and practice.

A Context
---------

Much recent work in statistical modeling and estimation has had the goal of producing sparse results and/or efficient, near automatic model selection. This objective is shared with other related practices–such as Deep Learning and Compressed Sensing. In the former case, we can point to Dropout [@srivastava_dropout_2014] and–in the latter–$\ell_p$ regularization [@donoho_compressed_2006] as basic examples.

Here we’ll simply assume that a practitioner intends to produce sparse estimates using the well-known LASSO–or $\ell_1$ penalty.

In PyMC3 [@salvatier_probabilistic_2016], the Bayes version of LASSO [@park_bayesian_2008] is easily specified.

``` python
import numpy as np
import scipy as sc

import pymc3 as pm
import theano
import theano.tensor as tt
from theano import shared as tt_shared

theano.config.mode = 'FAST_COMPILE'

mu_true = np.zeros(100)
mu_true[:20] = np.exp(-np.arange(20)) * 100

X = np.random.randn(int(np.alen(mu_true) * 0.7), np.alen(mu_true))
y = sc.stats.norm.rvs(loc=X.dot(mu_true), scale=10)

X_tt = tt_shared(X, name='X', borrow=True)
y_tt = tt_shared(y, name='y', borrow=True)

with pm.Model() as lasso_model:
    # Would be nice if we could pass the symbolic y_tt.shape, so
    # that our model would automatically conform to changes in
    # the shared variables X_tt.
    # See https://github.com/pymc-devs/pymc3/pull/1125
    beta_rv = pm.Laplace('beta', mu=0, b=1, shape=X.shape[1])
    y_rv = pm.Normal('y', mu=X_tt.dot(beta_rv), sd=1,
                     shape=y.shape[0], observed=y_tt)
```

Again, the negative total log likelihood in our example has a non-smooth $\ell_1$ term. Keeping this in mind, let’s say we wanted to produce a MAP estimate using PyMC3. A function is already provided for this task: `find_MAP`.

``` python
with lasso_model:
    params_0 = pm.find_MAP(vars=[beta_rv])
```

In our run of the above, an exception was thrown due to `nan` values within the gradient evaluation. We can inspect the gradient at $\beta = 0, 1$ and reproduce the result.

``` python
start = pm.Point({'beta': np.zeros(X.shape[1])}, model=lasso_model)
bij = pm.DictToArrayBijection(pm.ArrayOrdering(lasso_model.vars),
start)
logp = bij.mapf(lasso_model.fastlogp)
dlogp = bij.mapf(lasso_model.fastdlogp(lasso_model.vars))

# Could also inspect the log likelihood of the prior:
# beta_rv.dlogp().f(np.zeros_like(start['beta']))

grad_at_0 = dlogp(np.zeros_like(start['beta']))
grad_at_1 = dlogp(np.ones_like(start['beta']))
```

``` python
>>> print(np.sum(np.isnan(grad_at_0)))
100
>>> print(np.sum(np.isnan(grad_at_1)))
0
```

The s are not due to any short-coming of PyMC3; they only demonstrate a suitable place for our ideas and improvements. Additionally, by working within PyMC3, we can readily apply certain mathematical results. For instance, theorems that apply only to distributions. This idea is more relevant to the graph optimizations we consider later, but is still very important.

The Proximal Context
====================

We start with the essential ingredient: the proximal operator.

<div class="Def" markdown="" env-number="1" title-name="[Proximal Operator]">

$$\begin{equation*}
\operatorname*{prox}_{\phi}(x) =
    \operatorname*{argmin}_{z} \left\{
    \frac{1}{2} \left(z - x\right)^2 + \phi(z)
    \right\}
    \;.
\end{equation*}$$

</div>

As we mentioned earlier, the proximal operator is the main tool of proximal algorithms. Exact solutions to proximal operators exist for many $\phi$, and, since they’re often quite simple in form, their computation is relatively cheap: a property that the proximal methods themselves can inherit.

Consider the MAP estimation of a penalized likelihood, i.e. $$\begin{equation}
\beta^* = \operatorname*{argmin}_\beta \left\{ l(\beta) + \gamma \phi(\beta) \right\}
  \;,
  \label{eq:prox_problem}
\end{equation}$$ where functions $l$ and $\phi$ are commonly referred to as likelihood and prior terms (or loss and penalty), respectively. The proximal framework usually assumes $l$ and $\phi$ are at least lower semi-continuous and convex–although quite a few useful results still hold for non-convex functions.

Notice that Equation $\eqref{eq:prox_problem}$ takes the form of a proximal operator when $l(\beta) = \frac{1}{2} (y - \beta)^2$. Otherwise, in regression problems, we have $l(\beta) = \frac{1}{2} \|y - X \beta\|^2$. In this case, properties of the proximal operator can be used to produce independent proximal operators in each dimension of $\beta$. Since more than one property of the proximal operator can accomplish this–and result in distinct approaches–one might begin to see here a reason for the breadth of proximal methods.

The proximal operator relevant to our example, $\operatorname*{prox}_{|\cdot|}$, is equivalent to the soft-thresholding operator. Its implementation in Theano is somewhat trivial, but–for the sake of exposition–we provide an example.

``` python
beta_tt = tt.vector('beta', dtype=tt.config.floatX)
beta_tt.tag.test_value = np.r_[-10, -1, -0.2, 0, 0.2, 1,
10].astype(tt.config.floatX)

lambda_tt = tt.scalar('lambda', dtype=tt.config.floatX)
lambda_tt.tag.test_value = np.array(0.5).astype(tt.config.floatX)

def soft_threshold(beta_, lambda_):
    return tt.sgn(beta_) * tt.maximum(tt.abs_(beta_) - lambda_, 0)
```

``` python
>>> print(soft_threshold(beta_tt, lambda_tt).tag.test_value)
[-9.5 -0.5 -0.   0.   0.   0.5  9.5]
```

Proximal operators can be composed with a gradient step to produce the *proximal gradient* algorithm: $$\begin{equation}
\beta = \operatorname*{prox}_{\alpha \lambda \phi}(\beta - \alpha \nabla l(\beta))
  \;.
  \label{eq:forward-backward}
\end{equation}$$

Besides the proximal operator for $\phi$, steps in the proximal gradient algorithm are very straightforward and require only the gradient of $l(\beta)$. This is where a tangible benefit of symbolic computation becomes apparent: $\nabla l(\beta)$ can be computed automatically and efficiently. With \[backtracking\] line search to handle unknown step sizes, $\alpha$, the proximal gradient algorithm provides a surprisingly general means of sparse estimation.

The Symbolic Operations
=======================

In order to identify a relevant, non-smooth problem, check that a given proximal method’s conditions are satisfied (e.g. convexity), and potentially solve the resulting proximal operators in closed-form, we need to obtain expressions for $l$ and $\phi$.

In some cases, we’re able to tease apart $l$ and $\phi$ using only the interface provided by PyMC3. Specifically, the *observed* and *unobserved* random variable fields in PyMC3 models.

``` python
from theano import clone as tt_clone

logl = tt_clone(lasso_model.observed_RVs[0].logpt,
                {beta_rv: beta_tt})
logl.name = "logl"
```

Instead, let’s assume we’re extending `find_MAP` with even more generality, so that we can’t determine $l$ and $\phi$ in this way. This situation can occur when a user specifies custom distributions or potential functions. Regardless, we need to operate at a more symbolic level.

<div class="remark" markdown="" env-number="1" title-name="">

At this point, it is extremely worthwhile to browse the [Theano documentation](http://deeplearning.net/software/theano/extending/graphstructures.html) regarding graphs and their constituent objects.

</div>

The total log-likelihood is a good place to start. Let’s look at the symbolic graph for the log-likelihood of our model.

``` python
from theano import pp as tt_pp
from theano import pprint as tt_pprint
```

``` python
>>> print(tt_pp(lasso_model.logpt))
(Sum{acc_dtype=float64}(Sum{acc_dtype=float64}(((-log(TensorConstant{2}))
- (|(\beta - TensorConstant{0})| / TensorConstant{1})))) +
Sum{acc_dtype=float64}(Sum{acc_dtype=float64}(switch(TensorConstant{1},
(((TensorConstant{-1.0} * ((y - (X \dot \beta)) ** TensorConstant{2}))
+ log(TensorConstant{0.159154943092})) / TensorConstant{2.0}),
TensorConstant{-inf}))))
```

The [pretty printed](http://deeplearning.net/software/theano/tutorial/printing_drawing.html#pretty-printing) Theano graph tells us–among other things–that we indeed have a sum of $\ell_2$ and $\ell_1$ terms, although they are found among other confusing results (such as a `switch` statement).

As with most graphs produced by symbolic algebra systems, we need to understand how operations and objects are expressed in a graph and exactly which ones are relevant to us. After doing so, we can develop a means of finding what we want. The [debug printout](http://deeplearning.net/software/theano/tutorial/printing_drawing.html#debug-print) is often a better visual summary of graphs, since it expresses branches clearly.

``` python
>>> tt.printing.debugprint(lasso_model.logpt)
Elemwise{add,no_inplace} [id A] ''
 |Sum{acc_dtype=float64} [id B] ''
 | |Sum{acc_dtype=float64} [id C] ''
 |   |Elemwise{sub,no_inplace} [id D] ''
 |     |DimShuffle{x} [id E] ''
 |     | |Elemwise{neg,no_inplace} [id F] ''
 |     |   |Elemwise{log,no_inplace} [id G] ''
 |     |     |TensorConstant{2} [id H]
 |     |Elemwise{true_div,no_inplace} [id I] ''
 |       |Elemwise{abs_,no_inplace} [id J] ''
 |       | |Elemwise{sub,no_inplace} [id K] ''
 |       |   |beta [id L]
 |       |   |DimShuffle{x} [id M] ''
 |       |     |TensorConstant{0} [id N]
 |       |DimShuffle{x} [id O] ''
 |         |TensorConstant{1} [id P]
 |Sum{acc_dtype=float64} [id Q] ''
   |Sum{acc_dtype=float64} [id R] ''
     |Elemwise{switch,no_inplace} [id S] ''
       |DimShuffle{x} [id T] ''
       | |TensorConstant{1} [id P]
       |Elemwise{true_div,no_inplace} [id U] ''
       | |Elemwise{add,no_inplace} [id V] ''
       | | |Elemwise{mul,no_inplace} [id W] ''
       | | | |DimShuffle{x} [id X] ''
       | | | | |TensorConstant{-1.0} [id Y]
       | | | |Elemwise{pow,no_inplace} [id Z] ''
       | | |   |Elemwise{sub,no_inplace} [id BA] ''
       | | |   | |y [id BB]
       | | |   | |dot [id BC] ''
       | | |   |   |X [id BD]
       | | |   |   |beta [id L]
       | | |   |DimShuffle{x} [id BE] ''
       | | |     |TensorConstant{2} [id H]
       | | |DimShuffle{x} [id BF] ''
       | |   |Elemwise{log,no_inplace} [id BG] ''
       | |     |TensorConstant{0.159154943092} [id BH]
       | |DimShuffle{x} [id BI] ''
       |   |TensorConstant{2.0} [id BJ]
       |DimShuffle{x} [id BK] ''
         |TensorConstant{-inf} [id BL]
```

We see that the top-most operator is an `Elemwise` that applies the scalar `add` operation. This is the “$+$” in $l + \phi$. If we were to consider the inputs of this operator as candidates for $l$ and $\phi$, then we could do the following:

``` python
>>> print(lasso_model.logpt.owner.inputs)
[Sum{acc_dtype=float64}.0, Sum{acc_dtype=float64}.0]
```

Starting from the sub-graphs of each term, we could then search for any non-smooth functions that have known closed-form proximal operators. In our case, we only consider the absolute value function.

``` python
def get_abs_between(input_node):
    """ Search for `abs` in the operations between our input and the
    log-likelihood output node.
    """
    term_ops = list(tt.gof.graph.ops([input_node],
[lasso_model.logpt]))

    # Is there an absolute value in there?
    return filter(lambda x: x.op is tt.abs_, term_ops)

abs_res = [(get_abs_between(in_), in_)
           for in_ in lasso_model.logpt.owner.inputs]

for r_ in abs_res:
    if len(r_[0]) == 0:
        phi = r_[1]
    else:
        logp = r_[1]
```

``` python
>>> tt.printing.debugprint(logp)
Sum{acc_dtype=float64} [id A] ''
 |Sum{acc_dtype=float64} [id B] ''
   |Elemwise{switch,no_inplace} [id C] ''
     |DimShuffle{x} [id D] ''
     | |TensorConstant{1} [id E]
     |Elemwise{true_div,no_inplace} [id F] ''
     | |Elemwise{add,no_inplace} [id G] ''
     | | |Elemwise{mul,no_inplace} [id H] ''
     | | | |DimShuffle{x} [id I] ''
     | | | | |TensorConstant{-1.0} [id J]
     | | | |Elemwise{pow,no_inplace} [id K] ''
     | | |   |Elemwise{sub,no_inplace} [id L] ''
     | | |   | |y [id M]
     | | |   | |dot [id N] ''
     | | |   |   |X [id O]
     | | |   |   |beta [id P]
     | | |   |DimShuffle{x} [id Q] ''
     | | |     |TensorConstant{2} [id R]
     | | |DimShuffle{x} [id S] ''
     | |   |Elemwise{log,no_inplace} [id T] ''
     | |     |TensorConstant{0.159154943092} [id U]
     | |DimShuffle{x} [id V] ''
     |   |TensorConstant{2.0} [id W]
     |DimShuffle{x} [id X] ''
       |TensorConstant{-inf} [id Y]
>>> tt.printing.debugprint(phi)
Sum{acc_dtype=float64} [id A] ''
 |Sum{acc_dtype=float64} [id B] ''
   |Elemwise{sub,no_inplace} [id C] ''
     |DimShuffle{x} [id D] ''
     | |Elemwise{neg,no_inplace} [id E] ''
     |   |Elemwise{log,no_inplace} [id F] ''
     |     |TensorConstant{2} [id G]
     |Elemwise{true_div,no_inplace} [id H] ''
       |Elemwise{abs_,no_inplace} [id I] ''
       | |Elemwise{sub,no_inplace} [id J] ''
       |   |beta [id K]
       |   |DimShuffle{x} [id L] ''
       |     |TensorConstant{0} [id M]
       |DimShuffle{x} [id N] ''
         |TensorConstant{1} [id O]
```

The above approach is still too limiting; we need something more robust. For instance, our logic could fail on graphs that are expressed as $\eta (l + \phi) + 1$–although a graph for the equivalent expression $\eta l + \eta \phi + \eta$ might succeed. These are types of weaknesses inherent to naive approaches like ours. Furthermore, sufficient logic that uses a similar approach is likely to result in complicated and less approachable code.

The appropriate computational tools are found in the subjects of graph unification and term rewriting, as well as the areas of functional and logic programming. Luckily, Theano provides some basic unification capabilities through its `PatternSub` class.

`PatternSub` works within the context of Theano [graph optimization](http://deeplearning.net/software/theano/optimizations.html). Graph optimizations perform the common symbolic operations of reduction/simplification and rewriting. Consider the `phi` variable; the print-outs show an unnecessary subtraction with $0$. Clearly this step is unnecessary, so–in a basic way–we can see that the graph hasn’t been simplified, yet.

Many standard algebraic simplifications are already present in Theano, and, by creating our own graph optimizations, we can provide the advanced functionality we’ve been alluding to.

<div class="example" markdown="" env-number="1" title-name="[Algebraic Graph Optimization]">

As a quick demonstration, we’ll make replacement patterns for multiplicative distribution across two forms of addition: `sum` and `add`.

``` python
test_a_tt = tt.as_tensor_variable(5, name='a')
test_b_tt = tt.as_tensor_variable(2, name='b')
test_c_tt = tt.as_tensor_variable(np.r_[1, 2], name='c')

test_exprs_tt = (test_a_tt * test_b_tt,)
test_exprs_tt += (test_a_tt * (test_b_tt + test_a_tt),)
test_exprs_tt += (test_a_tt * (test_c_tt + test_a_tt),)
test_exprs_tt += (test_a_tt * (test_c_tt + test_c_tt),)

mul_dist_pat_tt = (tt.gof.opt.PatternSub(
    (tt.mul, 'x', (tt.sum, 'y', 'z')),
    (tt.sum, (tt.mul, 'x', 'y'), (tt.mul, 'x', 'z'))
),)
mul_dist_pat_tt += (tt.gof.opt.PatternSub(
    (tt.mul, 'x', (tt.add, 'y', 'z')),
    (tt.add, (tt.mul, 'x', 'y'), (tt.mul, 'x', 'z'))
),)
```

Substitutions can be applied to an objective function until it is in a fully-reduced form: `EquilibriumOptimizer` provides this functionality.

``` python
test_sub_eqz_opt_tt = tt.gof.opt.EquilibriumOptimizer(
    mul_dist_pat_tt, max_use_ratio=10)

test_fgraph_tt = tt.gof.fg.FunctionGraph(
    tt.gof.graph.inputs(test_exprs_tt), test_exprs_tt)
```

``` python
>>> tt.printing.debugprint(test_fgraph_tt)
Elemwise{mul,no_inplace} [id A] ''   5
 |TensorConstant{5} [id B]
 |TensorConstant{2} [id C]
Elemwise{mul,no_inplace} [id D] ''   8
 |TensorConstant{5} [id B]
 |Elemwise{add,no_inplace} [id E] ''   4
   |TensorConstant{2} [id C]
   |TensorConstant{5} [id B]
Elemwise{mul,no_inplace} [id F] ''   9
 |DimShuffle{x} [id G] ''   3
 | |TensorConstant{5} [id B]
 |Elemwise{add,no_inplace} [id H] ''   7
   |TensorConstant{[1 2]} [id I]
   |DimShuffle{x} [id J] ''   2
     |TensorConstant{5} [id B]
Elemwise{mul,no_inplace} [id K] ''   6
 |DimShuffle{x} [id L] ''   1
 | |TensorConstant{5} [id B]
 |Elemwise{add,no_inplace} [id M] ''   0
   |TensorConstant{[1 2]} [id I]
   |TensorConstant{[1 2]} [id I]
```

Now, when we apply the optimization, the `FunctionGraph` should contain the replacements.

``` python
test_fgraph_opt = test_sub_eqz_opt_tt.optimize(test_fgraph_tt)
```

``` python
>>> tt.printing.debugprint(test_fgraph_tt)
Elemwise{mul,no_inplace} [id A] ''   5
 |TensorConstant{5} [id B]
 |TensorConstant{2} [id C]
Elemwise{add,no_inplace} [id D] ''   10
 |Elemwise{mul,no_inplace} [id E] ''   4
 | |TensorConstant{5} [id B]
 | |TensorConstant{2} [id C]
 |Elemwise{mul,no_inplace} [id F] ''   3
   |TensorConstant{5} [id B]
   |TensorConstant{5} [id B]
Elemwise{add,no_inplace} [id G] ''   12
 |Elemwise{mul,no_inplace} [id H] ''   9
 | |DimShuffle{x} [id I] ''   2
 | | |TensorConstant{5} [id B]
 | |TensorConstant{[1 2]} [id J]
 |Elemwise{mul,no_inplace} [id K] ''   8
   |DimShuffle{x} [id I] ''   2
   |DimShuffle{x} [id L] ''   1
     |TensorConstant{5} [id B]
Elemwise{add,no_inplace} [id M] ''   11
 |Elemwise{mul,no_inplace} [id N] ''   7
 | |DimShuffle{x} [id O] ''   0
 | | |TensorConstant{5} [id B]
 | |TensorConstant{[1 2]} [id J]
 |Elemwise{mul,no_inplace} [id P] ''   6
   |DimShuffle{x} [id O] ''   0
   |TensorConstant{[1 2]} [id J]
```

</div>

Even more symbolic capabilities might be needed to \[efficiently\] achieve the functionality we desire. Standalone libraries like SymPy and [LogPy](https://github.com/logpy/logpy/) can be adapted to Theano graphs and provide these capabilities–although direct implementation in Theano may be better.

Finally, let’s briefly imagine how convexity could be determined symbolically. For differentiable terms, we could start with a simple second derivative test. Within Theano, a “second derivative” can be obtained using the `hessian` function, and within `theano.sandbox.linalg` are `Optimizer` hints for matrix positivity and other properties relevant to determining convexity.

<div class="remark" markdown="" env-number="2" title-name="">

Other great examples of linear algebra themed optimizations are in `theano.sandbox.linalg`: for instance, `no_transpose_symmetric`. Some of these demonstrate exactly how straight-forward adding algebraic features can be.

</div>

Although our convexity testing idea is far too simple for some functions, the point is that the basic tools necessary for work in this direction are already in place. With the logic programming and symbolic libraries mentioned earlier, a robust implementation of the convex function calculus could be very much in reach.

Discussion
==========

We’ve sketched out some ideas and tools with which one could develop a robust estimation platform guided by the more abstract mathematical frameworks from which new and efficient methods are produced.

Some key steps may require the integration of a fully featured symbolic algebra system. Along these lines, connections between Theano, SymPy and LogPy have been explored in @rocklin_mathematically_2013–as well as many other important aspects of the topics discussed here.

Besides the automation of proximal algorithms themselves, there are areas of application involving very large and complex models–perhaps the ones arising in Deep Learning. How might we consider the operator splitting of ADMM within deeply layered or hierarchical models [@polson_statistical_2015]? At which levels and on which terms should the splitting be performed? Beyond trying to solve the potentially unwieldy mathematics arising from such questions, by imbuing these symbolic tools with more mathematical awareness, we can at least experiment in these directions and quickly offer numerical solutions. This is–in part–the edge from which statistics hasn’t been benefiting and modern machine learning has.

Before closing, a very related–and interesting–set of ideas is worth mentioning: the possibility of encoding more symbolic knowledge into probabilistic programming platforms like PyMC3. Using the same optimization mechanisms as the examples here, simple distributional relationships can be encoded. For instance, the convolution of normally distributed random variables:

``` python
mu_X = tt.vector('mu_X')
mu_X.tag.test_value = np.array([1.], dtype=tt.config.floatX)
sd_X = tt.vector('sd_X')
sd_X.tag.test_value = np.array([2.], dtype=tt.config.floatX)

mu_Y = tt.vector('mu_Y')
mu_Y.tag.test_value = np.array([1.], dtype=tt.config.floatX)
sd_Y = tt.vector('sd_Y')
sd_Y.tag.test_value = np.array([0.5], dtype=tt.config.floatX)

with pm.Model() as conv_model:
    X_rv = pm.Normal('X', mu_X, sd=sd_X, shape=(1,))
    Y_rv = pm.Normal('Y', mu_Y, sd=sd_Y, shape=(1,))
    Z_rv = X_rv + Y_rv
```

We create a Theano `Op` to handle the convolution.

``` python
class NormConvOp(tt.Op):
    __props__ = ()

    def make_node(self, *inputs):
        name_new = str.join('+', [getattr(in_, 'name', '') for in_ in
inputs])
        mu_new = tt.add(*[in_.distribution.mu for in_ in inputs])
        sd_new = tt.sqrt(tt.add(*[in_.distribution.sd**2 for in_ in
inputs]))
        conv_rv = pm.Normal(name_new, mu=mu_new, sd=sd_new,
                            # Is this another place where
automatically/Theano managed
                            # shapes are really needed.  For now, we
hack it.
                            shape=(1,))

        return tt.Apply(self, inputs, [conv_rv])

    def perform(self, node, inputs, output_storage):
        z = output_storage[0]
        z[0] = np.add(*inputs)
```

Now, all that’s needed is a `PatternSub` like before.

``` python
def is_normal_dist(x):
    return hasattr(x, 'distribution') and isinstance(x.distribution,
pm.Normal)

norm_conv_pat_tt = (tt.gof.opt.PatternSub(
    (tt.add,
     {'pattern': 'x',
      'constraint': lambda x: is_normal_dist(x)},
     {'pattern': 'y',
      'constraint': lambda x: is_normal_dist(x)}
     ),
    (NormConvOp(), 'x', 'y')),)

norm_conv_opt_tt = tt.gof.opt.EquilibriumOptimizer(norm_conv_pat_tt,
                                                   max_use_ratio=10)

Z_fgraph_tt = tt.gof.fg.FunctionGraph([X_rv, Y_rv], [Z_rv])

# We lose the `FreeRV.distribution` attribute when cloning the graph
# with `theano.gof.graph.clone_get_equiv` in `FunctionGraph`, so this
# hackishly reattaches that information:
_ = [setattr(g_in, 'distribution', s_in.distribution)
     for s_in, g_in in zip([X_rv, Y_rv], Z_fgraph_tt.inputs)]
```

``` python
with conv_model:
    _ = norm_conv_opt_tt.optimize(Z_fgraph_tt)

norm_conv_var_dist = Z_fgraph_tt.outputs[0].distribution
```

The resulting graph:

``` python
>>> tt.printing.debugprint(Z_fgraph_tt)
NormConvOp [id A] 'X+Y'   0
 |X [id B]
 |Y [id C]
```

and the convolution’s parameters (for the test values):

``` python
>>> print(norm_conv_var_dist.mu.tag.test_value)
[ 2.]
>>> print(norm_conv_var_dist.sd.tag.test_value)
[ 2.06155281]
```

More sophisticated routines–like the example above–could implement parameter expansions, efficient re-parameterizations and equivalent scale mixture forms in an effort to optimize a graph for sampling or point evaluation. Objectives for these optimizations could be straightforward and computationally based (e.g. reducing the number of operations in computations of the log likelihood and other quantities) or more statistically focused (e.g. highly efficient sampling, improve mixing). These ideas are most definitely not new–one example is given by @mohasel_afshar_probabilistic_2016 for symbolic Gibbs sampling, but we hope the examples given here make the point that the tools are readily available and quite accessible.

We’ll end on a much more spacey consideration. Namely, that this is a context in which we can start experimenting rapidly with objectives over the space of estimation routines. This space is generated by–but not limited to–the variety of symbolic representations, re-parameterizations, etc., mentioned above. It does not necessarily require the complete estimation of a model at each step, nor even the numeric value of quantities like the gradient or Hessian. It may involve them, but not their evaluation; perhaps, instead, symbolic comparisons of competing gradients and Hessians arising from different representations. What we’re describing lies somewhere between the completely numeric assessments common today, and the entirely symbolic work found within the theorems and manipulations of the mathematics we use to derive methods.
