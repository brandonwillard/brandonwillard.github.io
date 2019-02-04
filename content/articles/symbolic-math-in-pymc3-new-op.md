---
bibliography:
- 'tex/symbolic-pymc3.bib'
modified: '2019-2-4'
tags: 'pymc3,theano,statistics,symbolic computation,python,probability theory'
title: Random Variables in Theano
date: '2018-12-28'
author: 'Brandon T. Willard'
figure_dir: '{attach}/articles/figures/'
figure_ext: png
---

<div class="abstract">
Continuing from <a href="#24875a2c31fa7f94ce562adddedc0bf8">Willard, Brandon T. (2018)</a>, we'll attempt to improve upon `RandomFunction` and make a case for a similar `Op` in PyMC3.

</div>


# Introduction

We'll call the new `Op` developed here `RandomVariable`, since random variables are the abstraction we're primarily targeting. `RandomVariable` will provide the functionality of `Distribution`, `FreeRV` and `ObservedRV`, and, by working at the `Op` level, it will be much more capable of leveraging existing Theano functionality.

Specifically, by using the `Op` interface, we're able to do the following:

1.  Remove the need for an explicitly specified shape parameter.

    <div class="example" markdown="">
    For example, definitions like

    ```{#org9f36fc6 .python}
    with pm.Model():
        X_rv = pm.Normal('X_rv', mu_X, sd=sd_X, shape=(1,))
    ```

    reduce to

    ```{#orgf62e7a4 .python}
    with pm.Model():
        X_rv = pm.Normal('X_rv', mu_X, sd=sd_X)
    ```

    </div>
2.  Random variable nodes created by an `Op` automatically implement `Distribution.default`/`Distribution.get_test_val` functionality and remove the reliance on initial values during random variable instantiation. `Op` automatically uses `Op.perform`, which will draw a sample as a test value **and** propagate it throughout the graph to down-stream tensor variables.
3.  Log-densities can be generated as secondary outputs of `Op.make_node`, which removes the need for `Distribution.logp*` methods.
4.  `pymc.distribution.draw_values` and related methods are no longer necessary; their functionality is already covered within Theano's existing graph machinery&#x2013;in the same way as `pymc.distribution.Distribution.default/get_test_val`.

The main points of entry in our `Op`, are `Op.make_node` and `Op.perform`. `Op.make_node` is used during symbolic graph creation and provides immediate access to the `Op`'s symbolic inputs&#x2013;serving a purpose similar to `Distribution.__init__`. `Op.make_node` is where shape inference tasks (e.g. [PyMC3 PR 1125](https://github.com/pymc-devs/pymc3/pull/1125)) are more suitably addressed; however, `Op` provides additional means of shape inference and management (e.g. `Op.infer_shape`) occurring at different phases of graph compilation that aren't readily accessible outside of the `Op` framework.


# A **new** Random Variable `Op`

```{#org9c584cc .python}
import sys
import os

from pprint import pprint

import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import theano
import theano.tensor as tt

theano.config.mode = 'FAST_COMPILE'
theano.config.exception_verbosity = 'high'
# NOTE: pymc3 requires test values
theano.config.compute_test_value = 'warn'

import pymc3 as pm
```

Most of the work involved in generalizing `RandomFunction` has to do with symbolic shape handling and inference. We need to bridge the gaps between symbolic array/tensor broadcasting parameters and the way Numpy random variable functions allow distribution parameters to be specified.

<div class="example" markdown="">
Scalar normal random variates have a support and parameters with dimension zero. In Listing [4](#org352d751) we create a scalar normal random variate in Numpy and inspect its shape. The length of the shape corresponds to the dimension of the distribution's support (i.e. zero).

```{#org352d751 .python}
np.shape(np.random.normal(loc=0, scale=1, size=None))
```

```{#org96612e2 .python}
()
```

Numpy also allows one to specify **independent** normal variates using one function call with each variate's parameters spanning dimensions higher than the variate's. In Listing [6](#orgbabaa48) we specify three independent scalar normal variates, each with a different mean and scale parameter. This time, the result's shape reflects **the number of independent random variates**, and not the dimension of the underlying distribution's support.

```{#orgbabaa48 .python}
np.shape(np.random.normal(loc=[0, 1, 2], scale=[1, 2, 3], size=None))
```

```{#orgef69d1e .python}
(3,)
```

Distribution parameters can also be broadcasted, as in [8](#org6643ee7). Now, each independent variate has the same scale value.

```{#org6643ee7 .python}
np.shape(np.random.normal(loc=[0, 1, 2], scale=1, size=None))
```

The `size` parameter effectively replicates variates, in-line with the&#x2013;potentially broadcasted&#x2013;distribution parameters.

When bridging these Numpy functions and Theano, we have to adapt the underlying parameter/shape logic of functions like `np.random.normal` to a scenario involving symbolic parameters and their symbolic shapes.

For instance, in Theano a **symbolic** scalar's shape is represented in nearly the same way.

```{#orga6116d4 .python}
test_scalar = tt.scalar()
test_scalar.shape.eval({test_scalar: 1})
```

```{#orgb9018eb .python}
[]
```

This means that our proposed Theano adaptation of `np.random.normal`, let's call it `tt_normal`, should return the same result as Numpy in the case of scalars.

What about `tt_normal(loc=tt.vector(), scale=tt.vector(), size=None)`? Since the inputs are purely symbolic, the resulting symbolic object's shape should be, too, but we should also know that the symbolic shape should have dimension equal to one. Just as in Listing [6](#orgbabaa48), each corresponding element in the vector arguments of `tt_normal` is an independent variate; in the symbolic case, we might not know exactly how many of them there are, yet, but we know that there's a vector's worth of them.

How exactly do we get that information from Theano, though? The type produced by `tt.vector` has an `ndim` parameter that provides this. Furthermore, there is some (intermittent) functionality that allows one to iterate over shapes. Listing [11](#org0d70518) demonstrates this.

```{#org0d70518 .python}
test_matrix = tt.matrix()
shape_parts = tuple(test_matrix.shape)
shape_parts
```

```{#orgd865db6 .python}
(Subtensor{int64}.0, Subtensor{int64}.0)
```

When the matrix in Listing [11](#org0d70518) is "materialized" (i.e. given a value), its corresponding shape object&#x2013;and its components&#x2013;will take their respective values.

```{#org4d51e03 .python}
tuple(p.eval({test_matrix: np.diag([1, 2])}) for p in shape_parts)
```

```{#orgc53dece .python}
(array(2), array(2))
```

If we knew that the support of this distribution was a scalar/vector/matrix, then these `ndim`-related results&#x2013;obtained from the symbolic parameters&#x2013;would tell us that we have multiple, independent variates and we could reliably extract the symbolic variables corresponding to those actual dimension sizes.

</div>

To determine the shape parts (i.e. support, number of independent and replicated variates) of the symbolic random variables, we mimic the corresponding Numpy logic and use the Theano `ndim` shape information described above. The following function generalizes that work for many simple distributions.

```{#org03297c0 .python}
from collections.abc import Iterable, ByteString
from warnings import warn
from copy import copy

from theano.tensor.raw_random import (RandomFunction, RandomStateType,
                                      _infer_ndim_bcast)


def param_supp_shape_fn(ndim_supp, ndims_params, dist_params,
                        rep_param_idx=0, param_shapes=None):
    """A function for deriving a random variable's support shape/dimensions
    from one of its parameters.

    XXX: It's not always possible to determine a random variable's support
    shape from its parameters, so this function has fundamentally limited
    applicability.

    XXX: This function is not expected to handle `ndim_supp = 0` (i.e.
    scalars), since that is already definitively handled in the `Op` that
    calls this.

    TODO: Consider using `theano.compile.ops.shape_i` alongside `ShapeFeature`.

    Parameters
    ==========
    ndim_supp: int
        Total number of dimensions in the support (assumedly > 0).
    ndims_params: list of int
        Number of dimensions for each distribution parameter.
    dist_params: list of `theano.gof.graph.Variable`
        The distribution parameters.
    param_shapes: list of `theano.compile.ops.Shape` (optional)
        Symbolic shapes for each distribution parameter.
        Providing this value prevents us from reproducing the requisite
        `theano.compile.ops.Shape` object (e.g. when it's already available to
        the caller).
    rep_param_idx: int (optional)
        The index of the distribution parameter to use as a reference
        In other words, a parameter in `dist_param` with a shape corresponding
        to the support's shape.
        The default is the first parameter (i.e. the value 0).

    Results
    =======
    out: a tuple representing the support shape for a distribution with the
    given `dist_params`.
    """
    # XXX: Gotta be careful slicing Theano variables, the `Subtensor` Op isn't
    # handled by `tensor.get_scalar_constant_value`!
    # E.g.
    #     test_val = tt.as_tensor_variable([[1], [4]])
    #     tt.get_scalar_constant_value(test_val.shape[-1]) # works
    #     tt.get_scalar_constant_value(test_val.shape[0]) # doesn't
    #     tt.get_scalar_constant_value(test_val.shape[:-1]) # doesn't
    if param_shapes is not None:
        # return param_shapes[0][-self.ndim_supp:]
        return (param_shapes[rep_param_idx][-ndim_supp],)
    else:
        # return dist_params[rep_param_idx].shape[-ndim_supp]
        ref_shape = tt.shape(dist_params[rep_param_idx])
        return (ref_shape[-ndim_supp],)
```

Finally, we put everything together in a new random variable `Op` called `RandomVariable`.

```{#orgb7517e4 .python}
class RandomVariable(tt.gof.Op):
    """This is essentially `RandomFunction`, except that it removes the `outtype`
    dependency and handles shape dimension information more directly.
    """
    __props__ = ('name', 'dtype', 'ndim_supp', 'inplace', 'ndims_params')

    def __init__(self, name, dtype, ndim_supp, ndims_params, rng_fn,
                 *args,
                 supp_shape_fn=param_supp_shape_fn,
                 inplace=False,
                 **kwargs):
        """Create a random variable `Op`.

        Parameters
        ==========
        name: str
            The `Op`'s display name.
        dtype: Theano dtype
            The underlying dtype.
        ndim_supp: int
            Dimension of the support.  This value is used to infer the exact
            shape of the support and independent terms from ``dist_params``.
        ndims_params: tuple (int)
            Number of dimensions of each parameter in ``dist_params``.
        rng_fn: function or str
            The non-symbolic random variate sampling function.
            Can be the string name of a method provided by
            `numpy.random.RandomState`.
        supp_shape_fn: callable (optional)
            Function used to determine the exact shape of the distribution's
            support.

            It must take arguments ndim_supp, ndims_params, dist_params
            (i.e. an collection of the distribution parameters) and an
            optional param_shapes (i.e. tuples containing the size of each
            dimension for each distribution parameter).

            Defaults to `param_supp_shape_fn`.
        inplace: boolean
            Determine whether or not the underlying rng state is updated in-place or
            not (i.e. copied).
        """
        super().__init__(*args, **kwargs)

        self.name = name
        self.ndim_supp = ndim_supp
        self.dtype = dtype
        self.supp_shape_fn = supp_shape_fn
        self.inplace = inplace

        if not isinstance(ndims_params, Iterable):
            raise ValueError('Parameter ndims_params must be iterable.')

        self.ndims_params = tuple(ndims_params)

        self.default_output = 1

        if isinstance(rng_fn, (str, ByteString)):
            self.rng_fn = getattr(np.random.RandomState, rng_fn)
        else:
            self.rng_fn = rng_fn

    def __str__(self):
        return '{}_rv'.format(self.name)

    def _infer_shape(self, size, dist_params, param_shapes=None):
        """Compute shapes and broadcasts properties.

        Inspired by `tt.add.get_output_info`.
        """

        size_len = tt.get_vector_length(size)

        dummy_params = tuple(p if n == 0 else tt.ones(tuple(p.shape)[:-n])
                             for p, n in zip(dist_params, self.ndims_params))

        _, out_bcasts, bcastd_inputs = tt.add.get_output_info(
            tt.DimShuffle, *dummy_params)

        # _, out_bcasts, bcastd_inputs = tt.add.get_output_info(tt.DimShuffle, *dist_params)

        bcast_ind, = out_bcasts
        ndim_ind = len(bcast_ind)
        shape_ind = bcastd_inputs[0].shape

        if self.ndim_supp == 0:
            shape_supp = tuple()

            # In the scalar case, `size` corresponds to the entire result's
            # shape. This implies the following:
            #     shape_ind[-ndim_ind] == size[:ndim_ind]
            # TODO: How do we add this constraint/check symbolically?

            ndim_reps = max(size_len - ndim_ind, 0)
            shape_reps = tuple(size)[ndim_ind:]
        else:
            shape_supp = self.supp_shape_fn(self.ndim_supp,
                                            self.ndims_params,
                                            dist_params,
                                            param_shapes=param_shapes)

            ndim_reps = size_len
            shape_reps = size

        ndim_shape = self.ndim_supp + ndim_ind + ndim_reps

        if ndim_shape == 0:
            shape = tt.constant([], dtype='int64')
        else:
            shape = tuple(shape_reps) + tuple(shape_ind) + tuple(shape_supp)

        # if shape is None:
        #     raise tt.ShapeError()

        return shape

    def compute_bcast(self, dist_params, size):
        """Compute the broadcast array for this distribution's `TensorType`.

        Parameters
        ==========
        dist_params: list
            Distribution parameters.
        size: int or Iterable (optional)
            Numpy-like size of the output (i.e. replications).
        """
        shape = self._infer_shape(size, dist_params)

        # Let's try to do a better job than `_infer_ndim_bcast` when
        # dimension sizes are symbolic.
        bcast = []
        for s in shape:
            try:
                if isinstance(s.owner.op, tt.Subtensor) and \
                   s.owner.inputs[0].owner is not None:
                    # Handle a special case in which
                    # `tensor.get_scalar_constant_value` doesn't really work.
                    s_x, s_idx = s.owner.inputs
                    s_idx = tt.get_scalar_constant_value(s_idx)
                    if isinstance(s_x.owner.op, tt.Shape):
                        x_obj, = s_x.owner.inputs
                        s_val = x_obj.type.broadcastable[s_idx]
                    else:
                        # TODO: Could go for an existing broadcastable here, too, no?
                        s_val = False
                else:
                    s_val = tt.get_scalar_constant_value(s)
            except tt.NotScalarConstantError:
                s_val = False

            bcast += [s_val == 1]
        return bcast

    def infer_shape(self, node, input_shapes):
        size = node.inputs[-2]
        dist_params = tuple(node.inputs[:-2])
        shape = self._infer_shape(size, dist_params,
                                  param_shapes=input_shapes[:-2])

        return [None, [s for s in shape]]

    def make_node(self, *dist_params, size=None, rng=None, name=None):
        """Create a random variable node.

        XXX: Unnamed/non-keyword arguments are considered distribution
        parameters!  If you want to set `size`, `rng`, and/or `name`, use their
        keywords.

        Parameters
        ==========
        dist_params: list
            Distribution parameters.
        size: int or Iterable (optional)
            Numpy-like size of the output (i.e. replications).
        rng: RandomState (optional)
            Existing Theano `RandomState` object to be used.  Creates a
            new one, if `None`.
        name: str (optional)
            Label for the resulting node.

        Results
        =======
        out: `Apply`
            A node with inputs `dist_args + (size, in_rng, name)` and outputs
            `(out_rng, sample_tensorvar)`.
        """
        if size is None:
            size = tt.constant([], dtype='int64')
        elif isinstance(size, int):
            size = tt.as_tensor_variable([size], ndim=1)
        elif not isinstance(size, Iterable):
            raise ValueError('Parameter size must be None, int, or an iterable with ints.')
        else:
            size = tt.as_tensor_variable(size, ndim=1)

        assert size.dtype in tt.int_dtypes

        dist_params = tuple(tt.as_tensor_variable(p)
                            for p in dist_params)

        if rng is None:
            rng = theano.shared(np.random.RandomState())
        elif not isinstance(rng.type, RandomStateType):
            warn('The type of rng should be an instance of RandomStateType')

        bcast = self.compute_bcast(dist_params, size)

        # dtype = tt.scal.upcast(self.dtype, *[p.dtype for p in dist_params])

        outtype = tt.TensorType(dtype=self.dtype, broadcastable=bcast)
        out_var = outtype(name=name)
        inputs = dist_params + (size, rng)
        outputs = (rng.type(), out_var)

        return theano.gof.Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        """Draw samples using Numpy/SciPy."""
        rng_out, smpl_out = outputs

        # Draw from `rng` if `self.inplace` is `True`, and from a copy of `rng`
        # otherwise.
        args = list(inputs)
        rng = args.pop()
        size = args.pop()

        assert isinstance(rng, np.random.RandomState), (type(rng), rng)

        rng_out[0] = rng

        # The symbolic output variable corresponding to value produced here.
        out_var = node.outputs[1]

        # If `size == []`, that means no size is enforced, and NumPy is
        # trusted to draw the appropriate number of samples, NumPy uses
        # `size=None` to represent that.  Otherwise, NumPy expects a tuple.
        if np.size(size) == 0:
            size = None
        else:
            size = tuple(size)

        if not self.inplace:
            rng = copy(rng)

        smpl_val = self.rng_fn(rng, *(args + [size]))

        if (not isinstance(smpl_val, np.ndarray) or
            str(smpl_val.dtype) != out_var.type.dtype):
            smpl_val = theano._asarray(smpl_val, dtype=out_var.type.dtype)

        # When `size` is `None`, NumPy has a tendency to unexpectedly
        # return a scalar instead of a higher-dimension array containing
        # only one element. This value should be reshaped
        # TODO: Really?  Why shouldn't the output correctly correspond to
        # the returned NumPy value?  Sounds more like a mis-specification of
        # the symbolic output variable.
        if size is None and smpl_val.ndim == 0 and out_var.ndim > 0:
            smpl_val = smpl_val.reshape([1] * out_var.ndim)

        smpl_out[0] = smpl_val

    def grad(self, inputs, outputs):
        return [theano.gradient.grad_undefined(self, k, inp,
                                               'No gradient defined through raw random numbers op')
                for k, inp in enumerate(inputs)]

    def R_op(self, inputs, eval_points):
        return [None for i in eval_points]
```


# Using `RandomVariable`

In Listing [17](#orgf494cec) we create some `RandomVariable` `Op`s.

```{#orgf494cec .python}
import scipy
from functools import partial


# Continuous Numpy-generated variates
class UniformRVType(RandomVariable):
    def __init__(self):
        super().__init__('uniform', theano.config.floatX, 0, [0, 0], 'uniform', inplace=True)

    def make_node(self, lower, upper, size=None, rng=None, name=None):
        return super().make_node(lower, upper, size=size, rng=rng, name=name)

UniformRV = UniformRVType()


class NormalRVType(RandomVariable):
    def __init__(self):
        super().__init__('normal', theano.config.floatX, 0, [0, 0], 'normal', inplace=True)

    def make_node(self, mu, sigma, size=None, rng=None, name=None):
        return super().make_node(mu, sigma, size=size, rng=rng, name=name)


NormalRV = NormalRVType()


class GammaRVType(RandomVariable):
    def __init__(self):
        super().__init__('gamma', theano.config.floatX, 0, [0, 0], 'gamma', inplace=True)

    def make_node(self, shape, scale, size=None, rng=None, name=None):
        return super().make_node(shape, scale, size=size, rng=rng, name=name)


GammaRV = GammaRVType()


class ExponentialRVType(RandomVariable):
    def __init__(self):
        super().__init__('exponential', theano.config.floatX, 0, [0], 'exponential', inplace=True)

    def make_node(self, scale, size=None, rng=None, name=None):
        return super().make_node(scale, size=size, rng=rng, name=name)


ExponentialRV = ExponentialRVType()


# One with multivariate support
class MvNormalRVType(RandomVariable):
    def __init__(self):
        super().__init__('multivariate_normal', theano.config.floatX, 1, [1, 2], 'multivariate_normal', inplace=True)

    def make_node(self, mean, cov, size=None, rng=None, name=None):
        return super().make_node(mean, cov, size=size, rng=rng, name=name)


MvNormalRV = MvNormalRVType()


class DirichletRVType(RandomVariable):
    def __init__(self):
        super().__init__('dirichlet', theano.config.floatX, 1, [1], 'dirichlet', inplace=True)

    def make_node(self, alpha, size=None, rng=None, name=None):
        return super().make_node(alpha, size=size, rng=rng, name=name)


DirichletRV = DirichletRVType()


# A discrete Numpy-generated variate
class PoissonRVType(RandomVariable):
    def __init__(self):
        super().__init__('poisson', 'int64', 0, [0], 'poisson', inplace=True)

    def make_node(self, rate, size=None, rng=None, name=None):
        return super().make_node(rate, size=size, rng=rng, name=name)


PoissonRV = PoissonRVType()


# A SciPy-generated variate
class CauchyRVType(RandomVariable):
    def __init__(self):
        super().__init__('cauchy', theano.config.floatX, 0, [0, 0],
                         lambda rng, *args: scipy.stats.cauchy.rvs(*args, random_state=rng),
                         inplace=True)

    def make_node(self, loc, scale, size=None, rng=None, name=None):
        return super().make_node(loc, scale, size=size, rng=rng, name=name)


CauchyRV = CauchyRVType()


# Support shape is determined by the first dimension in the *second* parameter (i.e.
# the probabilities vector)
class MultinomialRVType(RandomVariable):
    def __init__(self):
        super().__init__('multinomial', 'int64', 1, [0, 1], 'multinomial',
                         supp_shape_fn=partial(param_supp_shape_fn, rep_param_idx=1),
                         inplace=True)

    def make_node(self, n, pvals, size=None, rng=None, name=None):
        return super().make_node(n, pvals, size=size, rng=rng, name=name)


MultinomialRV = MultinomialRVType()
```

<div class="example" markdown="">
In Listing [18](#orged59f09) we draw samples from instances of `RandomVariable`s.

```{#orged59f09 .python}
print("UniformRV(0., 30., size=[10]):\n{}\n".format(
    UniformRV(0., 30., size=[10]).eval()
))

print("NormalRV([0., 100.], 30, size=[4, 2]):\n{}\n".format(
    NormalRV([0., 100.], 30, size=[4, 2]).eval()))

print("GammaRV([2., 1.], 2., size=[4, 2]):\n{}\n".format(
    GammaRV([2., 1.], 2., size=[4, 2]).eval()))

print("ExponentialRV([2., 50.], size=[4, 2]):\n{}\n".format(
    ExponentialRV([2., 50.], size=[4, 2]).eval()))

print("MvNormalRV([0, 1e2, 2e3], np.diag([1, 1, 1]), size=[3, 2, 3]):\n{}\n".format(
    MvNormalRV([0, 1e2, 2e3], np.diag([1, 1, 1]), size=[2, 3]).eval()))

print("DirichletRV([0.1, 10, 0.5], size=[3, 2, 3]):\n{}\n".format(
    DirichletRV([0.1, 10, 0.5], size=[2, 3]).eval()))

print("PoissonRV([2., 1.], size=[4, 2]):\n{}\n".format(
    PoissonRV([2., 15.], size=[4, 2]).eval()))

print("CauchyRV([1., 100.], 30, size=[4, 2]):\n{}\n".format(
    CauchyRV([1., 100.], 30, size=[4, 2]).eval()))

print("MultinomialRV(20, [1/6.]*6, size=[6, 2]):\n{}".format(
    MultinomialRV(20, [1 / 6.] * 6, size=[3, 2]).eval()))
```

```{#orga89dd83 .python}
UniformRV(0., 30., size=[10]):
[ 5.83131933 28.56231204 20.73018065 17.21042461 25.53140341 23.76268637
 28.27629994  7.10457399 19.88378878 26.62382369]

NormalRV([0., 100.], 30, size=[4, 2]):
[[  0.73277898  98.26041204]
 [-25.9810085   79.13385495]
 [-23.17013683 130.86966242]
 [-52.83756722  95.21829178]]

GammaRV([2., 1.], 2., size=[4, 2]):
[[5.09679154 0.6149213 ]
 [2.64231927 0.7277265 ]
 [5.98877316 0.41751667]
 [3.77525439 1.11561567]]

ExponentialRV([2., 50.], size=[4, 2]):
[[ 2.29684191  7.12084933]
 [ 0.39386731 38.79158981]
 [ 1.11400165  4.31175303]
 [ 1.50499115  9.65667649]]

MvNormalRV([0, 1e2, 2e3], np.diag([1, 1, 1]), size=[3, 2, 3]):
[[[-6.67447019e-01  9.88636435e+01  1.99973471e+03]
  [ 6.06351715e-01  9.96429347e+01  1.99915978e+03]
  [ 1.12246741e+00  9.96807860e+01  2.00201859e+03]]

 [[ 3.61931404e-02  9.89907880e+01  2.00036910e+03]
  [-1.61077330e+00  1.01905479e+02  2.00134565e+03]
  [ 9.45854243e-01  1.00877071e+02  1.99914438e+03]]]

DirichletRV([0.1, 10, 0.5], size=[3, 2, 3]):
[[[1.41863953e-06 9.35392908e-01 6.46056738e-02]
  [4.50961569e-15 9.71338820e-01 2.86611803e-02]
  [2.41299980e-05 9.94566812e-01 5.40905817e-03]]

 [[5.79850503e-08 9.73090671e-01 2.69092713e-02]
  [4.17758767e-09 9.61671733e-01 3.83282630e-02]
  [8.78921782e-03 9.54146972e-01 3.70638103e-02]]]

PoissonRV([2., 1.], size=[4, 2]):
[[ 1 15]
 [ 1 12]
 [ 2 21]
 [ 1 14]]

CauchyRV([1., 100.], 30, size=[4, 2]):
[[ -86.93222925   79.9758127 ]
 [  13.41882831 -374.41779179]
 [  75.74505567   93.2944822 ]
 [  30.0824262   130.40873511]]

MultinomialRV(20, [1/6.]*6, size=[6, 2]):
[[[2 4 4 2 4 4]
  [2 5 2 4 3 4]]

 [[2 5 6 2 4 1]
  [0 4 4 3 5 4]]

 [[6 1 1 4 4 4]
  [3 4 3 2 3 5]]]


```

</div>

As noted, there are a few long-standing difficulties surrounding the use and determination of shape information in PyMC3. `RandomVariable` doesn't suffer the same limitations.

<div class="example" markdown="">
In Listing [20](#org77fdfa2), we see that a multivariate normal random variable cannot be created in PyMC3 without explicit shape information.

```{#org77fdfa2 .python}
import traceback

test_mean = tt.vector('test_mean')
test_cov = tt.matrix('test_cov', dtype='int64')

test_mean.tag.test_value = np.asarray([1])
test_cov.tag.test_value = np.asarray([[1]])

try:
  with pm.Model():
    test_rv = pm.MvNormal('test_rv', test_mean, test_cov)
except Exception as e:
  print("".join(traceback.format_exception_only(type(e), e)))
```

```{#org9a55c25 .python}
ValueError: Invalid dimension for value: 0


```

As Listing [22](#orgb7414f6) demonstrates, the same construction is possible when one specifies an explicit size/shape.

```{#orgb7414f6 .python}
try:
  with pm.Model():
    test_rv = pm.MvNormal('test_rv', test_mean, test_cov, shape=1)
    print("test_rv.distribution.shape = {}".format(test_rv.distribution.shape))
    print("test_rv.tag.test_value = {}".format(test_rv.tag.test_value))
except Exception as e:
  print("".join(traceback.format_exception_only(type(e), e)))
```

```{#orga8cc629 .python}
test_rv.distribution.shape = [1]
test_rv.tag.test_value = [1.]


```

</div>

Using `RandomVariable`, we do not have to specify a shape, nor implement any sampling code outside of `RandomVariable.perform` to draw random variables and generate valid test values.

<div class="example" markdown="">
Listings [24](#org67b2727) and [26](#org56bde38) demonstrate how easy it is to create dependencies between random variates using `RandomVariable`, and how sampling and test values are automatic. It uses a multivariate normal as the mean of another multivariate normal.

```{#org67b2727 .python}
theano.config.compute_test_value = 'ignore'

mu_tt = tt.vector('mu')
C_tt = tt.matrix('C')
D_tt = tt.matrix('D')

X_rv = MvNormalRV(mu_tt, C_tt)
Y_rv = MvNormalRV(X_rv, D_tt)

# Sample some values under specific parameter values
print("{} ~ X\n{} ~ Y".format(
    X_rv.eval({mu_tt: [1, 2], C_tt: np.diag([1, 2])}),
    Y_rv.eval({mu_tt: [1, 2], C_tt: np.diag([1, 2]), D_tt: np.diag([10, 20])})))
```

```{#orgd1cac3d .python}
[-1.25047147  4.87459955] ~ X
[ 2.15486205 -3.3066946 ] ~ Y


```

```{#org56bde38 .python}
theano.config.compute_test_value = 'warn'

mu_tt.tag.test_value = np.array([0, 30, 40])
C_tt.tag.test_value = np.diag([100, 10, 1])
D_tt.tag.test_value = np.diag([100, 10, 1])

X_rv = MvNormalRV(mu_tt, C_tt)
Y_rv = MvNormalRV(X_rv, D_tt)

# Observe the automatically generated test values
print("X test value: {}\nY test value: {}".format(
    X_rv.tag.test_value,
    Y_rv.tag.test_value))

```

```{#orgecacfb5 .python}
X test value: [ 1.78826967 28.73266332 38.57297111]
Y test value: [33.93703352 27.48925582 38.21563854]


```

</div>

<div class="example" markdown="">
In Listing [28](#orge489ad8), we specify the following hierarchical model:

\begin{equation*}
  \begin{aligned}
    M &\sim \text{Poisson}\left(10\right)
    \\
    \alpha_i &\sim \text{Uniform}\left(0, 1\right),
    \quad i \in \left\{0, \dots, M\right\}
    \\
    \pi &\sim \text{Dirichlet}\left(\alpha\right)
    \\
    Y &\sim \text{Multinomial}\left(M, \pi\right)
  \end{aligned}
  \;.
\end{equation*}

This toy model is particularly interesting in how it specifies symbolic dependencies between continuous and discrete distributions and uses random variables to determine the shapes of other random variables.

```{#orge489ad8 .python}
theano.config.compute_test_value = 'ignore'
pois_rate = tt.dscalar('rate')
test_pois_rv = PoissonRV(pois_rate)
test_alpha = UniformRV(0, 1, size=test_pois_rv)
test_dirichlet_rv = DirichletRV(test_uniform_rv)
test_multinom_rv = MultinomialRV(test_pois_rv, test_dirichlet_rv)

test_multinom_draw = theano.function(inputs=[], outputs=test_multinom_rv,
                                     givens={pois_rate: 10.})

print("test_multinom_rv draw 1: {}\ntest_multinom_rv draw 2: {}".format(
    test_multinom_draw(), test_multinom_draw()))
```

```{#orgff8bd09 .python}
test_multinom_rv draw 1: [0 2 0 0 1 0 2 1 0 0]
test_multinom_rv draw 2: [5 2 1 0 0 0 1 0 1 1 0 1 0]


```

</div>


## Random Variable Pretty Printing

In Listing [30](#org52ecc27), we implement a pretty printer that produces more readable forms of Theano graphs containing `RandomVariable` nodes.

```{#org52ecc27 .python}
class RandomVariablePrinter:
    """Pretty print random variables.
    """
    def __init__(self, name=None):
        """
        Parameters
        ==========
        name: str (optional)
            A fixed name to use for the random variables printed by this
            printer.  If not specified, use `RandomVariable.name`.
        """
        self.name = name

    def process_param(self, idx, sform, pstate):
        """Special per-parameter post-formatting.

        This can be used, for instance, to change a std. dev. into a variance.

        Parameters
        ==========
        idx: int
            The index value of the parameter.
        sform: str
            The pre-formatted string form of the parameter.
        pstate: object
            The printer state.
        """
        return sform

    def process(self, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        pprinter = pstate.pprinter
        node = output.owner

        if node is None or not isinstance(node.op, RandomVariable):
            raise TypeError("function %s cannot represent a variable that is "
                            "not the result of a RandomVariable operation" %
                            self.name)

        new_precedence = -1000
        try:
            old_precedence = getattr(pstate, 'precedence', None)
            pstate.precedence = new_precedence
            out_name = VariableWithShapePrinter.process_variable_name(
                output, pstate)
            shape_info_str = VariableWithShapePrinter.process_shape_info(
                output, pstate)
            if getattr(pstate, 'latex', False):
                dist_format = "%s \\sim \\operatorname{%s}\\left(%s\\right)"
                dist_format += ', \\quad {}'.format(shape_info_str)
            else:
                dist_format = "%s ~ %s(%s)"
                dist_format += ',  {}'.format(shape_info_str)

            op_name = self.name or node.op.name
            dist_params = node.inputs[:-2]
            formatted_params = [
                self.process_param(i, pprinter.process(p, pstate), pstate)
                for i, p in enumerate(dist_params)
            ]

            dist_params_r = dist_format % (out_name,
                                           op_name,
                                           ", ".join(formatted_params))
        finally:
            pstate.precedence = old_precedence

        pstate.preamble_lines += [dist_params_r]
        pstate.memo[output] = out_name

        return out_name
```

```{#org414cfc6 .python}
import string

from copy import copy
from collections import OrderedDict

from sympy import Array as SympyArray
from sympy.printing import latex as sympy_latex


class VariableWithShapePrinter:
    """Print variable shape info in the preamble and use readable character
    names for unamed variables.
    """
    available_names = OrderedDict.fromkeys(string.ascii_letters)
    default_printer = theano.printing.default_printer

    @classmethod
    def process(cls, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        using_latex = getattr(pstate, 'latex', False)

        if isinstance(output, tt.gof.Constant):
            if output.ndim > 0 and using_latex:
                out_name = sympy_latex(SympyArray(output.data))
            else:
                out_name = str(output.data)
        elif isinstance(output, tt.TensorVariable):
            # Process name and shape
            out_name = cls.process_variable_name(output, pstate)
            shape_info = cls.process_shape_info(output, pstate)
            pstate.preamble_lines += [shape_info]
        elif output.name:
            out_name = output.name
        else:
            out_name = cls.default_printer.process(output, pstate)

        pstate.memo[output] = out_name
        return out_name

    @classmethod
    def process_shape_name(cls, output, pstate):
        shape_of_var = output.owner.inputs[0]
        shape_names = pstate.memo.setdefault('shape_names', {})
        out_name = shape_names.setdefault(
            shape_of_var, cls.process_variable_name(output, pstate))
        return out_name

    @classmethod
    def process_variable_name(cls, output, pstate):
        if output in pstate.memo:
            return pstate.memo[output]

        available_names = getattr(pstate, 'available_names', None)
        if available_names is None:
            # Initialize this state's available names
            available_names = copy(cls.available_names)
            fgraph = getattr(output, 'fgraph', None)
            if fgraph:
                # Remove known names in the graph.
                _ = [available_names.pop(v.name, None)
                     for v in fgraph.variables]
            setattr(pstate, 'available_names', available_names)

        if output.name:
            # Observed an existing name; remove it.
            out_name = output.name
            available_names.pop(out_name, None)
        else:
            # Take an unused name.
            out_name, _ = available_names.popitem(last=False)

        pstate.memo[output] = out_name
        return out_name

    @classmethod
    def process_shape_info(cls, output, pstate):
        using_latex = getattr(pstate, 'latex', False)

        if output.dtype in tt.int_dtypes:
            sspace_char = 'Z'
        elif output.dtype in tt.uint_dtypes:
            sspace_char = 'N'
        elif output.dtype in tt.float_dtypes:
            sspace_char = 'R'
        else:
            sspace_char = '?'

        fgraph = getattr(output, 'fgraph', None)
        shape_feature = None
        if fgraph:
            if not hasattr(fgraph, 'shape_feature'):
                fgraph.attach_feature(tt.opt.ShapeFeature())
            shape_feature = fgraph.shape_feature

        shape_dims = []
        for i in range(output.ndim):
            s_i_out = None
            if using_latex:
                s_i_pat = '{n^{%s}}' + ('_{%s}' % i)
            else:
                s_i_pat = 'n^%s' + ('_%s' % i)
            if shape_feature:
                new_precedence = -1000
                try:
                    old_precedence = getattr(pstate, 'precedence', None)
                    pstate.precedence = new_precedence
                    _s_i_out = shape_feature.get_shape(output, i)
                    if _s_i_out.owner:
                        if (isinstance(_s_i_out.owner.op, tt.Subtensor) and
                            all(isinstance(i, tt.Constant)
                                for i in _s_i_out.owner.inputs)):
                            s_i_out = str(_s_i_out.owner.inputs[0].data[
                                _s_i_out.owner.inputs[1].data])
                        elif not isinstance(_s_i_out, tt.TensorVariable):
                            s_i_out = pstate.pprinter.process(_s_i_out, pstate)
                except KeyError:
                    pass
                finally:
                    pstate.precedence = old_precedence

            if not s_i_out:
                s_i_out = cls.process_variable_name(output, pstate)
                s_i_out = s_i_pat % s_i_out

            shape_dims += [s_i_out]

        shape_info = cls.process_variable_name(output, pstate)
        if using_latex:
            shape_info += ' \\in \\mathbb{%s}' % sspace_char
            shape_dims = ' \\times '.join(shape_dims)
            if shape_dims:
                shape_info += '^{%s}' % shape_dims
        else:
            shape_info += ' in %s' % sspace_char
            shape_dims = ' x '.join(shape_dims)
            if shape_dims:
                shape_info += '**(%s)' % shape_dims

        return shape_info
```

```{#orgccd5273 .python}
import textwrap


class PreamblePPrinter(theano.printing.PPrinter):
    """Pretty printer that displays a preamble.

    For example,

        X ~ N(\mu, \sigma)
        (b * X)

    XXX: Not thread-safe!
    """
    def __init__(self, *args, pstate_defaults=None, **kwargs):
        """
        Parameters
        ==========
        pstate_defaults: dict (optional)
            Default printer state parameters.
        """
        super().__init__(*args, **kwargs)
        self.pstate_defaults = pstate_defaults or {}
        self.printers_dict = dict(tt.pprint.printers_dict)
        self.printers = copy(tt.pprint.printers)
        self._pstate = None

    def create_state(self, pstate):
        # FIXME: Find all the user-defined node names and make the tag
        # generator aware of them.
        if pstate is None:
            pstate = theano.printing.PrinterState(
                pprinter=self,
                preamble_lines=[],
                **self.pstate_defaults)
        elif isinstance(pstate, dict):
            pstate.setdefault('preamble_lines', [])
            pstate.update(self.pstate_defaults)
            pstate = theano.printing.PrinterState(pprinter=self, **pstate)

        # FIXME: Good old fashioned circular references...
        # We're doing this so that `self.process` will be called correctly
        # accross all code.  (I'm lookin' about you, `DimShufflePrinter`; get
        # your act together.)
        pstate.pprinter._pstate = pstate

        return pstate

    def process(self, r, pstate=None):
        pstate = self._pstate
        assert pstate
        return super().process(r, pstate)

    def process_graph(self, inputs, outputs, updates=None,
                      display_inputs=False):
        raise NotImplemented()

    def __call__(self, *args, latex_env='equation', latex_label=None):
        var = args[0]
        pstate = next(iter(args[1:]), None)
        if isinstance(pstate, (theano.printing.PrinterState, dict)):
            pstate = self.create_state(args[1])
        elif pstate is None:
            pstate = self.create_state(None)
        # else:
        #     # XXX: The graph processing doesn't pass around the printer state!
        #     # TODO: We'll have to copy the code and fix it...
        #     raise NotImplemented('No preambles for graph printing, yet.')

        # This pretty printer needs more information about shapes and inputs,
        # which it gets from a `FunctionGraph`.  Create one, if `var` isn't
        # already assigned one.
        fgraph = getattr(var, 'fgraph', None)
        if not fgraph:
            fgraph = tt.gof.fg.FunctionGraph(
                tt.gof.graph.inputs([var]), [var])
            var = fgraph.outputs[0]

            # Use this to get better shape info
            shape_feature = tt.opt.ShapeFeature()
            fgraph.attach_feature(shape_feature)

        body_str = super().__call__(var, pstate)

        latex_out = getattr(pstate, 'latex', False)
        if pstate.preamble_lines and latex_out:
            preamble_str = "\n\\\\\n".join(pstate.preamble_lines)
            preamble_str = "\\begin{gathered}\n%s\n\\end{gathered}" % (preamble_str)
            res = "\n\\\\\n".join([preamble_str, body_str])
        else:
            res = "\n".join(pstate.preamble_lines + [body_str])

        if latex_out and latex_env:
            label_out = f'\\label{{{latex_label}}}\n' if latex_label else ''
            res = textwrap.indent(res, '\t\t')
            res = (f"\\begin{{{latex_env}}}\n"
                   f"{res}\n"
                   f"{label_out}"
                   f"\\end{{{latex_env}}}")

        return res
```

```{#orgfc82717 .python}
tt_pprint = PreamblePPrinter()

tt_pprint.assign(lambda pstate, r: True, VariableWithShapePrinter)
tt_pprint.assign(UniformRV, RandomVariablePrinter('U'))
tt_pprint.assign(GammaRV, RandomVariablePrinter('Gamma'))
tt_pprint.assign(ExponentialRV, RandomVariablePrinter('Exp'))


class NormalRVPrinter(RandomVariablePrinter):
    def __init__(self):
        super().__init__('N')

    def process_param(self, idx, sform, pstate):
        if idx == 1:
            if getattr(pstate, 'latex', False):
                return f'{{{sform}}}^{{2}}'
            else:
                return f'{sform}**2'
        else:
            return sform

tt_pprint.assign(NormalRV, NormalRVPrinter())
tt_pprint.assign(MvNormalRV, NormalRVPrinter())

tt_pprint.assign(DirichletRV, RandomVariablePrinter('Dir'))
tt_pprint.assign(PoissonRV, RandomVariablePrinter('Pois'))
tt_pprint.assign(CauchyRV, RandomVariablePrinter('C'))
tt_pprint.assign(MultinomialRV, RandomVariablePrinter('MN'))

tt_tex_pprint = PreamblePPrinter(pstate_defaults={'latex': True})
tt_tex_pprint.printers = copy(tt_pprint.printers)
tt_tex_pprint.printers_dict = dict(tt_pprint.printers_dict)
tt_tex_pprint.assign(tt.mul, theano.printing.OperatorPrinter('\\odot', -1, 'either'))
tt_tex_pprint.assign(tt.true_div, theano.printing.PatternPrinter(('\\frac{%(0)s}{%(1)s}', -1000)))
tt_tex_pprint.assign(tt.pow, theano.printing.PatternPrinter(('{%(0)s}^{%(1)s}', -1000)))
```

<div class="example" markdown="">
Listing [35](#org1ed4a46), creates a graph with two random variables and prints the results with the default Theano pretty printer as Equation \(\eqref{eq:rv-pprinter-exa}\).

```{#org0e19f4e .python}


tt.config.compute_test_value = 'ignore'

Z_tt = UniformRV(tt.scalar('l_0'), tt.scalar('l_1'), name='Z')
X_tt = NormalRV(Z_tt, tt.scalar('\sigma_1'), name='X')
Y_tt = MvNormalRV(tt.vector('\mu'), tt.abs_(X_tt) * tt.constant(np.diag([1, 2])), name='Y')

W_tt = X_tt * (tt.scalar('b') * Y_tt + tt.scalar('c'))
```

```{#org1ed4a46 .python}
print(tt_tex_pprint(W_tt, latex_label='eq:rv-pprinter-exa'))
```

\begin{equation}
		\begin{gathered}
		l_0 \in \mathbb{R}
		\\
		l_1 \in \mathbb{R}
		\\
		Z \sim \operatorname{U}\left(l_0, l_1\right), \quad Z \in \mathbb{R}
		\\
		\sigma_1 \in \mathbb{R}
		\\
		X \sim \operatorname{N}\left(Z, {\sigma_1}^{2}\right), \quad X \in \mathbb{R}
		\\
		b \in \mathbb{R}
		\\
		\mu \in \mathbb{R}^{{n^{\mu}}_{0}}
		\\
		Y \sim \operatorname{N}\left(\mu, {(|X| \odot \left[\begin{matrix}1 & 0\\0 & 2\end{matrix}\right])}^{2}\right), \quad Y \in \mathbb{R}^{{n^{Y}}_{0}}
		\\
		c \in \mathbb{R}
		\end{gathered}
		\\
		(X \odot ((b \odot Y) + c))
\label{eq:rv-pprinter-exa}
\end{equation}

</div>


# Algebraic Manipulations

With our new `RandomVariable`, we can alter the replacement patterns used by `tt.gof.opt.PatternSub` in <a href="#24875a2c31fa7f94ce562adddedc0bf8">Willard, Brandon T. (2018)</a> and implement a slightly better parameter lifting for affine transforms of scalar normal random variables in Listing [36](#orgc483b75).

```{#orgc483b75 .python}
norm_lift_pats = [
    # Lift element-wise multiplication
    tt.gof.opt.PatternSub(
        (tt.mul,
         'a_x',
         (NormalRV, 'mu_x', 'sd_x', 'size_x', 'rs_x')),
        (NormalRV,
         (tt.mul, 'a_x', 'mu_x'),
         (tt.mul, 'a_x', 'sd_x'),
         'size_x',
         'rs_x',
        )),
    # Lift element-wise addition
    tt.gof.opt.PatternSub(
        (tt.add,
         (NormalRV, 'mu_x', 'sd_x', 'size_x', 'rs_x'),
         'b_x'),
        (NormalRV,
         (tt.add, 'mu_x', 'b_x'),
         'sd_x',
         'size_x',
         'rs_x',
        )),
]

norm_lift_opts = tt.gof.opt.EquilibriumOptimizer(
    norm_lift_pats, max_use_ratio=10)
```

<div class="example" markdown="">
```{#orgc69f52b .python}
# [[file:~/projects/websites/brandonwillard.github.io/content/articles/src/org/symbolic-math-in-pymc3-new-op.org::graph-manipulation-setup][graph-manipulation-setup]]
from theano.gof import FunctionGraph, Feature, NodeFinder
from theano.gof.graph import inputs as tt_inputs, clone_get_equiv

theano.config.compute_test_value = 'ignore'
# graph-manipulation-setup ends here

mu_X = tt.vector('\mu')
sd_X = tt.vector('\sigma')

a_tt = tt.fscalar('a')
b_tt = tt.fscalar('b')

X_rv = NormalRV(mu_X, sd_X, name='X')
trans_X_rv = a_tt * X_rv + b_tt

trans_X_graph = FunctionGraph(tt_inputs([trans_X_rv]), [trans_X_rv])

# Create a copy and optimize that
trans_X_graph_opt = trans_X_graph.clone()

_ = norm_lift_opts.optimize(trans_X_graph_opt)
```

```{#org2bd1258 .python}
print(tt_tex_pprint(trans_X_graph.outputs[0], latex_env='equation*'))
```

Before applying the optimization:

\begin{equation*}
		\begin{gathered}
		a \in \mathbb{R}
		\\
		\mu \in \mathbb{R}^{{n^{\mu}}_{0}}
		\\
		\sigma \in \mathbb{R}^{{n^{\sigma}}_{0}}
		\\
		X \sim \operatorname{N}\left(\mu, {\sigma}^{2}\right), \quad X \in \mathbb{R}^{{n^{X}}_{0}}
		\\
		b \in \mathbb{R}
		\end{gathered}
		\\
		((a \odot X) + b)
\end{equation*}

```{#orge71b0ac .python}
print(tt_tex_pprint(trans_X_graph_opt.outputs[0], latex_env='equation*'))
```

After applying the optimization:

\begin{equation*}
		\begin{gathered}
		a \in \mathbb{R}
		\\
		\mu \in \mathbb{R}^{{n^{\mu}}_{0}}
		\\
		b \in \mathbb{R}
		\\
		\sigma \in \mathbb{R}^{{n^{\sigma}}_{0}}
		\\
		c \sim \operatorname{N}\left(((a \odot \mu) + b), {(a \odot \sigma)}^{2}\right), \quad c \in \mathbb{R}^{{n^{c}}_{0}}
		\end{gathered}
		\\
		c
\end{equation*}

</div>

Now, what if we wanted to handle affine transformations of a multivariate normal random variable? Specifically, consider the following:

\begin{equation*}
  X \sim N\left(\mu, \Sigma \right), \quad
  A X \sim N\left(A \mu, A \Sigma A^\top \right)
 \;.
\end{equation*}

At first, the substitution pattern in Listing [40](#org4a792de) might seem reasonable.

```{#org4a792de .python}
# Vector multiplication
tt.gof.opt.PatternSub(
    (tt.dot, 'A_x',
     (MvNormalRV, 'mu_x', 'cov_x', 'size_x', 'rs_x')),
    (MvNormalRV,
     (tt.dot, 'A_x', 'mu_x'),
     (tt.dot,
      (tt.dot, 'A_x', 'cov_x')
      (tt.transpose, 'A_x')),
     'size_x',
     'rs_x',
    ))
```

Unfortunately, the combination of size parameter and broadcasting complicates the scenario. Both parameters indirectly affect the distribution parameters, making the un-lifted dot-product consistent, but not necessarily the lifted products.

The following example demonstrates the lifting issues brought on by broadcasting.

<div class="example" markdown="">
We create a simple multivariate normal in Listing [41](#org7446c7b).

```{#org7446c7b .python}
mu_X = [0, 10]
cov_X = np.diag([1, 1e-2])
size_X_rv = [2, 3]
X_rv = MvNormalRV(mu_X, cov_X, size=size_X_rv)

print('{} ~ X_rv\n'.format(X_rv.tag.test_value))
```

```{#org9e35ff4 .python}
[[[-0.68284424  9.95587926]
  [ 1.66236785  9.87590909]
  [ 0.23449772 10.12455681]]

 [[ 0.3342739  10.05580428]
  [-0.18913408 10.0359336 ]
  [-1.2463576   9.90671218]]] ~ X_rv


```

Next, we create a simple matrix operator to apply to the multivariate normal.

```{#org0f84661 .python}
A_tt = tt.as_tensor_variable([[2, 5, 8], [3, 4, 9]])
# or A_tt = tt.as_tensor_variable([[2, 5, 8]])

# It's really just `mu_X`...
E_X_rv = X_rv.owner.inputs[2]

print('A * X_rv =\n{}\n'.format(tt.dot(A_tt, X_rv).tag.test_value))
```

```{#org05ebd11 .python}
A * X_rv =
[[[  1.18524621 150.31045062]
  [  1.07000851 150.65771936]]

 [[  1.31685497 160.33572146]
  [  0.33506491 160.82202495]]]


```

As we can see, the multivariate normal's test/sampled value has the correct shape for our matrix operator.

```{#orgd95a139 .python}
import traceback
try:
    print('A * E[X_rv] =\n{}\n'.format(tt.dot(A_tt, E_X_rv).tag.test_value))
except ValueError as e:
    print("".join(traceback.format_exception_only(type(e), e)))
```

```{#org51f4e3a .python}
ValueError: shapes (2,3) and (2,) not aligned: 3 (dim 1) != 2 (dim 0)


```

However, we see that the multivariate normal's inputs (i.e. the `Op` inputs)&#x2013;specifically the mean parameter&#x2013;do not directly reflect the support's shape, as one might expect.

```{#org6e32649 .python}
size_tile = tuple(size_X_rv) + (1,)
E_X_rv_ = tt.tile(E_X_rv, size_tile, X_rv.ndim)

print('A * E[X_rv] =\n{}\n'.format(tt.dot(A_tt, E_X_rv_).tag.test_value))
```

```{#org7f64727 .python}
A * E[X_rv] =
[[[  0 150]
  [  0 150]]

 [[  0 160]
  [  0 160]]]


```

We can manually replicate the inputs so that they match the output shape, but a solution to the general problem requires a more organized response.

</div>


# A Problem with Conversion from PyMC3

As in <a href="#24875a2c31fa7f94ce562adddedc0bf8">Willard, Brandon T. (2018)</a>, we can create mappings between existing PyMC3 random variables and their new `RandomVariable` equivalents.

<div class="example" markdown="">
```{#org8bffc80 .python}
pymc_theano_rv_equivs = {
    pm.Normal:
    lambda dist, rand_state:
    (None,
     # PyMC3 shapes aren't NumPy-like size parameters, so we attempt to
     # adjust for that.
     NormalRV(dist.mu, dist.sd, size=dist.shape[1:], rng=rand_state)),
    pm.MvNormal:
    lambda dist, rand_state:
    (None, NormalRV(dist.mu, dist.cov, size=dist.shape[1:], rng=rand_state)),
}
```

</div>

However, if we attempt the same PymC3 graph conversion approach as before (i.e. convert a PyMC3 model to a Theano `FunctionGraph` using `model_graph`, then replace PyMC3 random variable nodes with our new random variable types using `create_theano_rvs`), we're likely to run into a problem involving mismatching broadcastable dimensions.

The problem arises because **PyMC3 "knows" more broadcast information than it should**, since it uses the Theano variables' test values in order to obtain concrete shapes for the random variables it creates. Using concrete, non-symbolic shapes, it can exactly determine what would otherwise be ambiguous [broadcastable dimensions](http://deeplearning.net/software/theano/library/tensor/basic.html?highlight=broadcastable#theano.tensor.TensorType.broadcastable) at the symbolic level.

More specifically, broadcast information is required during the construction of a Theano `TensorType`, so PyMC3 random variable types can be inconsistent (unnecessarily restrictive, really) causing Theano to complain when we try to construct a `FunctionGraph`.

<div class="example" markdown="">
Consider the following example; it constructs two purely symbolic Theano vectors: one with broadcasting and one without.

```{#orgbd54927 .python}
y_tt = tt.row('y')
print("y_tt.broadcastable = {}".format(y_tt.broadcastable))

x_tt = tt.matrix('x')
print("x_tt.broadcastable = {}".format(x_tt.broadcastable))
```

```{#orga8f9d4c .python}
y_tt.broadcastable = (True, False)
x_tt.broadcastable = (False, False)


```

Notice that it&#x2013;by default&#x2013;signifies no broadcasting on its first and only dimension.

If we wish&#x2013;or if [Theano's configuration demands](http://deeplearning.net/software/theano/library/config.html#config.compute_test_value) it&#x2013;we can assign the symbolic vector arbitrary test values, as long as they're consistent with its type (i.e. a vector, or 1-dimensional array).

In the following, we assign both a broadcastable (i.e. first&#x2013;and only&#x2013;dimension has size 1) and non-broadcastable test value.

Test value is broadcastable:

```{#orgaadc727 .python}
from contextlib import contextmanager


x_tt.tag.test_value = np.array([[5]])

print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(x_tt.tag.test_value).broadcastable))
print("x_tt.broadcastable = {}".format(x_tt.broadcastable))

@contextmanager
def short_exception_msg(exc_type):
    _verbosity = theano.config.exception_verbosity
    theano.config.exception_verbosity = 'low'
    try:
        yield
    except exc_type as e:
        import traceback
        print("".join(traceback.format_exception_only(type(e), e)))
    finally:
        theano.config.exception_verbosity = _verbosity


with short_exception_msg(TypeError):
    x_tt.shape
    print("shape checks out!")
```

```{#org9c02ec5 .python}
test_value.broadcastable = (True, True)
x_tt.broadcastable = (False, False)
shape checks out!


```

```{#org56323a0 .python}
y_tt.tag.test_value = np.array([[5]])

print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(y_tt.tag.test_value).broadcastable))
print("y_tt.broadcastable = {}".format(y_tt.broadcastable))

with short_exception_msg(TypeError):
    y_tt.shape
    print("shape checks out!")
```

```{#orga78b1b0 .python}
test_value.broadcastable = (True, True)
y_tt.broadcastable = (True, False)
shape checks out!


```

Test value is **not** broadcastable:

```{#orge2d0568 .python}
x_tt.tag.test_value = np.array([[5, 4]])
print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(x_tt.tag.test_value).broadcastable))
print("x_tt.broadcastable = {}".format(x_tt.broadcastable))

with short_exception_msg(TypeError):
    x_tt.shape
    print("shape checks out!")
```

```{#org3ac23d5 .python}
test_value.broadcastable = (True, False)
x_tt.broadcastable = (False, False)
shape checks out!


```

```{#org53e07ae .python}
y_tt.tag.test_value = np.array([[5, 4], [3, 2]])
print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(y_tt.tag.test_value).broadcastable))
print("y_tt.broadcastable = {}".format(y_tt.broadcastable))

with short_exception_msg(TypeError):
    y_tt.shape
    print("shape checks out!")
```

```{#org60e8d8f .python}
test_value.broadcastable = (False, False)
y_tt.broadcastable = (True, False)
TypeError: For compute_test_value, one input test value does not have the requested type.

Backtrace when that variable is created:

  File "/home/bwillard/apps/anaconda3/envs/github-website/lib/python3.6/site-packages/IPython/terminal/interactiveshell.py", line 485, in mainloop
    self.interact()
  File "/home/bwillard/apps/anaconda3/envs/github-website/lib/python3.6/site-packages/IPython/terminal/interactiveshell.py", line 476, in interact
    self.run_cell(code, store_history=True)
  File "/home/bwillard/apps/anaconda3/envs/github-website/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2662, in run_cell
    raw_cell, store_history, silent, shell_futures)
  File "/home/bwillard/apps/anaconda3/envs/github-website/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2785, in _run_cell
    interactivity=interactivity, compiler=compiler, result=result)
  File "/home/bwillard/apps/anaconda3/envs/github-website/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2909, in run_ast_nodes
    if self.run_code(code, result):
  File "/home/bwillard/apps/anaconda3/envs/github-website/lib/python3.6/site-packages/IPython/core/interactiveshell.py", line 2963, in run_code
    exec(code_obj, self.user_global_ns, self.user_ns)
  File "<ipython-input-19-7427b1688530>", line 1, in <module>
    __org_babel_python_fname = '/tmp/user/1000/babel-fsZXPU/python-cZypXi'; __org_babel_python_fh = open(__org_babel_python_fname); exec(compile(__org_babel_python_fh.read(), __org_babel_python_fname, 'exec')); __org_babel_python_fh.close()
  File "/tmp/user/1000/babel-fsZXPU/python-cZypXi", line 1, in <module>
    y_tt = tt.row('y')

The error when converting the test value to that variable type:
Non-unit value on shape on a broadcastable dimension.
(2, 2)
(True, False)


```

Simply put: non-broadcastable Theano tensor variable types can take broadcastable and non-broadcastable values, while broadcastable types can only take broadcastable values.

</div>

What we can take from the example above is that if we determine that a vector has broadcastable dimensions using test values&#x2013;as PyMC3 does&#x2013;we unnecessarily introduce restrictions and potential inconsistencies down the line. One point of origin for such issues is **shared variables**.


# Discussion

In follow-ups to this series, we'll address a few loose ends, such as

-   the inclusion of density functions and likelihoods,
-   decompositions/reductions of overlapping multivariate types (e.g. transforms between tensors of univariate normals and equivalent multivariate normals),
-   canonicalization of graphs containing `RandomVariable` terms,
-   and more optimizations that specifically target MCMC schemes (e.g. automatic conversion to scale mixture decompositions).
