---
bibliography:
- 'tex/symbolic-pymc3.bib'
modified: '2018-12-28'
tags: 'pymc3,theano,statistics,symbolic computation,python,probability theory'
title: Random Variables in Theano
date: '2018-12-28'
author: 'Brandon T. Willard'
figure_dir: '{attach}/articles/figures/'
figure_ext: png
---

<div class="abstract">
Continuing from <sup id="24875a2c31fa7f94ce562adddedc0bf8"><a href="#WillardSymbolicMathPyMC32018" title="@misc{WillardSymbolicMathPyMC32018, title = {Symbolic {{Math}} in {{PyMC3}}}, urldate = {2018-12-27}, url = {https://brandonwillard.github.io/symbolic-math-in-pymc3.html}, author = {Willard, Brandon T.}, month = dec, year = {2018}, file = {/home/bwillard/Zotero/storage/6VVT4UNF/symbolic-math-in-pymc3.html} }">WillardSymbolicMathPyMC32018</a></sup>, we'll attempt to improve upon `RandomFunction` and make a case for a similar `Op` in PyMC3.

</div>


# A **new** Random Variable `Op`

We'll call this new `Op` `RandomVariable`, since random variables are the abstraction we're primarily targeting. `RandomVariable` will provide the functionality of `Distribution`, `FreeRV` and `ObservedRV`, and, by working at the `Op` level, it will be much more capable of leveraging existing Theano functionality.

Specifically, by using the `Op` interface, we're able to do the following:

1.  Reduce/remove the need for an explicitly specified shape parameter.

    <div class="example" markdown="">

    For example, definitions like

    ```{.python}
    with pm.Model():
        X_rv = pm.Normal('X_rv', mu_X, sd=sd_X, shape=(1,))
    ```

    reduce to

    ```{.python}
    with pm.Model():
        X_rv = pm.Normal('X_rv', mu_X, sd=sd_X)
    ```

    </div>
2.  Random variable nodes created by an `Op` automatically implement `Distribution.default`/`Distribution.get_test_val` functionality and remove the reliance on initial values during random variable instantiation. `Op` automatically uses `Op.perform`, which will draw a sample as a test value **and** propagate it throughout the graph to derived/down-stream tensor variables.
3.  Log-densities can be generated as secondary outputs of `Op.make_node`, which removes the need for `Distribution.logp*` methods.
4.  `pymc.distribution.draw_values` and related methods are no longer necessary; their functionality is already covered within Theano's existing graph machinery&#x2013;in the same way as `pymc.distribution.Distribution.default/get_test_val`.

The main points of entry in our `Op`, are `Op.make_node` and `Op.perform`. `Op.make_node` is used during symbolic graph creation and provides immediate access to the `Op`'s symbolic inputs&#x2013;serving a purpose similar to `Distribution.__init__`. `Op.make_node` is where shape inference tasks (e.g. [PyMC3 PR 1125](https://github.com/pymc-devs/pymc3/pull/1125)) are more suitably addressed; however, `Op` provides additional means of shape inference and management (e.g. `Op.infer_shape`) occurring at different phases of graph compilation that aren't readily accessible outside of the `Op` framework.


## Implementation

```{#import_theano_pymc3 .python}
import sys
import os

from pprint import pprint

import numpy as np

os.environ['MKL_THREADING_LAYER'] = 'GNU'

import theano
import theano.tensor as tt

theano.config.mode = 'FAST_COMPILE'
theano.config.exception_verbosity = 'high'
theano.config.compute_test_value = 'raise'

import pymc3 as pm
```

```{#supp_shape_fn .python}
from collections.abc import Iterable, ByteString
from warnings import warn
from copy import copy

from theano.tensor.raw_random import (RandomFunction, RandomStateType,
                                      _infer_ndim_bcast)


def matched_supp_shape_fn(ndim_supp, ndims_params, dist_params,
                          param_shapes=None):
    """A function for extracting a random variable's support shape/dimensions
    from other (e.g. distribution parameters) shape information.

    This default implementation uses the first non-independent/replicated
    dimension of the first distribution parameter.

    For example, with a normal random variable, the shape of the mean parameter
    along the first dimension will be used; dimensions after that, if any,
    determine the mean parameters for other *independent* normal variables.
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
        return (param_shapes[0][-ndim_supp],)
    else:
        ref_shape = tt.shape(dist_params[0])
        # return ref_shape[-self.ndim_supp:]
        return (ref_shape[-ndim_supp],)
```

```{#new_rv_op .python}
class RandomVariable(tt.gof.Op):
    """This is essentially `RandomFunction`, except that it removes the `outtype`
    dependency and handles shape dimension information more directly.
    """
    __props__ = ('name', 'dtype', 'ndim_supp', 'inplace', 'ndims_params')

    def __init__(self, name, ndim_supp, ndims_params, rng_fn, *args,
                 supp_shape_fn=None, dtype=theano.config.floatX, inplace=False,
                 **kwargs):
        """Create a random variable `Op`.

        Parameters
        ==========
        ndim_supp: int
            Dimension of the support.  This value is used to infer the exact
            shape of the support and independent terms from ``dist_params``.
        ndims_params: tuple (int)
            Number of dimensions for each parameter in ``dist_params``
            for a single variate.  Used to determine the shape of the
            independent variate space.
        rng_fn: function or str
            Sampler function.  Can be the string name of a method provided by
            `numpy.random.RandomState`.
        supp_shape_fn: callable
            Function used to determine the exact shape of the distribution's
            support. It must take arguments ndim_supp, ndims_params,
            dist_params (i.e. an collection of the distribution parameters) and an
            optional param_shapes (i.e. tuples containing the size of each
            dimension for each distribution parameter).

            Defaults to `supp_shape_fn`
        """
        super().__init__(*args, **kwargs)

        self.name = name
        self.inplace = inplace
        self.dtype = dtype

        self.supp_shape_fn = supp_shape_fn or matched_supp_shape_fn

        self.ndim_supp = ndim_supp

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
        # .tag.test_value

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

    def make_node(self, *dist_params, size=None, rng=None, name=None):
        """This will be the "constructor" called by users.

        Parameters
        ==========
        dist_params: list
            Distribution parameters.
        size: Iterable or None
            Numpy-like size of the output (i.e. replications).
        rng: RandomState or None
            Existing Theano `RandomState` object to be used.  Creates a
            new one, if `None`.

        Results
        =======
        An `Apply` node with rng state and sample tensor outputs.
        """
        if not (size is None or isinstance(size, Iterable)):
            raise ValueError('Parameter size must be None or iterable')

        dist_params = tuple(tt.as_tensor_variable(p)
                            for p in dist_params)

        dtype = tt.scal.upcast(self.dtype, *[p.dtype for p in dist_params])

        if rng is None:
            rng = theano.shared(np.random.RandomState())
        elif not isinstance(rng.type, RandomStateType):
            warn('The type of rng should be an instance of RandomStateType')

        if size is None:
            size = tt.constant([], dtype='int64')
        else:
            size = tt.as_tensor_variable(size, ndim=1)

        assert size.dtype == 'int64'

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

        outtype = tt.TensorType(dtype=dtype, broadcastable=bcast)

        out_var = outtype(name=name)

        inputs = (rng, size) + dist_params
        outputs = (rng.type(), out_var)

        return theano.gof.Apply(self, inputs, outputs)

    def infer_shape(self, node, input_shapes):
        size = node.inputs[1]
        dist_params = tuple(node.inputs[2:])
        shape = self._infer_shape(size, dist_params,
                                  param_shapes=input_shapes[2:])

        return [None, [s for s in shape]]

    def perform(self, node, inputs, outputs):
        """Uses `self.rng_fn` to draw random numbers."""
        rng_out, smpl_out = outputs

        # Draw from `rng` if `self.inplace` is `True`, and from a
        # copy of `rng` otherwise.
        rng, size, args = inputs[0], inputs[1], inputs[2:]

        assert type(rng) == np.random.RandomState, (type(rng), rng)

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

<div class="example" markdown="">

Here are some examples of `RandomVariable` in action.

```{#random_variable_example .python}
NormalRV = RandomVariable('normal', 0, [0, 0], 'normal')
MvNormalRV = RandomVariable('multivariate_normal', 1, [1, 2], 'multivariate_normal')

print("NormalRV([0., 100.], 30, size=[4, 2]):\n{}\n".format(
    NormalRV([0., 100.], 30, size=[4, 2]).eval()))

print("MvNormalRV([0, 1e2, 2e3], np.diag([1, 1, 1]), size=[3, 2, 3]):\n{}".format(
    MvNormalRV([0, 1e2, 2e3], np.diag([1, 1, 1]), size=[3, 2, 3]).eval()))
```

```{.python}
NormalRV([0., 100.], 30, size=[4, 2]):
[[ 27.43632901  93.39441318]
 [ 15.85959218 133.70107347]
 [ -9.36097104  68.13953575]
 [ 13.41389074 102.36908493]]

MvNormalRV([0, 1e2, 2e3], np.diag([1, 1, 1]), size=[3, 2, 3]):
[[[[-1.95386446e+00  1.00395353e+02  2.00061885e+03]
   [ 1.05345467e-02  1.00488854e+02  2.00075391e+03]
   [-2.96094820e-01  9.97791115e+01  2.00039472e+03]]

  [[ 1.09725296e+00  1.00481658e+02  2.00168402e+03]
   [-5.97706814e-01  9.84007224e+01  1.99998250e+03]
   [ 6.05551705e-01  1.00905256e+02  1.99904533e+03]]]


 [[[ 3.13777265e-01  9.73407662e+01  1.99854117e+03]
   [-8.54227150e-01  1.01332597e+02  1.99945623e+03]
   [-4.71493054e-01  9.98134334e+01  2.00121111e+03]]

  [[-3.61516595e-01  9.99772061e+01  1.99853933e+03]
   [ 5.17271791e-01  9.74970520e+01  2.00052296e+03]
   [ 2.15904894e-01  9.99694385e+01  2.00021462e+03]]]


 [[[-9.61903721e-01  1.00569645e+02  1.99858826e+03]
   [-6.72526622e-01  1.00277177e+02  2.00036283e+03]
   [ 2.58608351e-01  1.01020520e+02  1.99866678e+03]]

  [[ 8.70002780e-02  1.01020608e+02  1.99975983e+03]
   [ 8.56671014e-01  9.97595247e+01  2.00094824e+03]
   [ 6.17426596e-01  1.01919972e+02  1.99914348e+03]]]]


```

</div>

As we've mentioned, there are a few difficulties surrounding the use and determination of shape information in PyMC3. `RandomVariable` doesn't suffer the same limitations.

<div class="example" markdown="" title-name="">

A multivariate normal random variable cannot be created without explicit shape information.

```{.python}
import traceback

test_mean = tt.vector('test_mean')
test_mean.tag.test_value = np.asarray([1])

test_cov = tt.matrix('test_cov')
test_cov.tag.test_value = np.asarray([[1]])

try:
  with pm.Model():
    test_rv = pm.MvNormal('test_rv', test_mean, test_cov)
except Exception as e:
  print("".join(traceback.format_exception_only(type(e), e)))
```

```{.python}
ValueError: Invalid dimension for value: 0


```

```{.python}
try:
  with pm.Model():
    test_rv = pm.MvNormal('test_rv', test_mean, test_cov, shape=1)
    print("test_rv.distribution.shape = {}".format(test_rv.distribution.shape))
    print("test_rv.tag.test_value = {}".format(test_rv.tag.test_value))
except Exception as e:
  print("".join(traceback.format_exception_only(type(e), e)))
```

```{.python}
test_rv.distribution.shape = [1]
test_rv.tag.test_value = [1.]


```

Using `RandomVariable`, we do not have to specify a shape, nor implement any sampling code outside of `RandomVariable.perform` to draw random variables and generate valid test values.

```{.python}
test_mv_rv = MvNormalRV(test_mean, test_cov)
test_mv_rv_2 = MvNormalRV(test_mv_rv, test_cov)

# Observe the automatically generated test values
print("test_mv_rv.tag.test_value = {}".format(test_mv_rv.tag.test_value))
print("test_mv_rv_2.tag.test_value = {}".format(test_mv_rv_2.tag.test_value))

# Sample some values under specific parameter values
print("test_mv_rv.eval() = {}".format(test_mv_rv.eval(
    {test_mean: [1, 2], test_cov: np.diag([1, 2])})))
print("test_mv_rv_2.eval() = {}".format(test_mv_rv_2.eval(
    {test_mean: [1, 2, 3], test_cov: np.diag([1, 2, 70])})))
```

```{.python}
test_mv_rv.tag.test_value = [0.90661799]
test_mv_rv_2.tag.test_value = [1.10201953]
test_mv_rv.eval() = [-1.90184227  1.8679379 ]
test_mv_rv_2.eval() = [ 2.05659828 -1.33638943  3.85355663]


```

</div>


# A Problem with PyMC3 Broadcast Dimensions

As in <sup id="24875a2c31fa7f94ce562adddedc0bf8"><a href="#WillardSymbolicMathPyMC32018" title="@misc{WillardSymbolicMathPyMC32018, title = {Symbolic {{Math}} in {{PyMC3}}}, urldate = {2018-12-27}, url = {https://brandonwillard.github.io/symbolic-math-in-pymc3.html}, author = {Willard, Brandon T.}, month = dec, year = {2018}, file = {/home/bwillard/Zotero/storage/6VVT4UNF/symbolic-math-in-pymc3.html} }">WillardSymbolicMathPyMC32018</a></sup>, we can create mappings between existing PyMC3 random variables and their new `RandomVariable` equivalents.

```{#pymc_theano_rv_equivs .python}
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

However, if we attempt the same PymC3 graph conversion approach as before (i.e. convert a PyMC3 model to a Theano `FunctionGraph` using `model_graph`, then replace PyMC3 random variable nodes with our new random variable types using `create_theano_rvs`), we're likely to run into a problem involving mismatching broadcastable dimensions.

The problem arises because **PyMC3 "knows" more broadcast information than it should**, since it uses the Theano variables' test values in order to obtain concrete shapes for the random variables it creates. Using concrete, non-symbolic shapes, it can exactly determine what would otherwise be ambiguous [broadcastable dimensions](http://deeplearning.net/software/theano/library/tensor/basic.html?highlight=broadcastable#theano.tensor.TensorType.broadcastable) at the symbolic level.

More specifically, broadcast information is required during the construction of a Theano `TensorType`, so PyMC3 random variable types can be inconsistent (unnecessarily restrictive, really) causing Theano to complain when we try to construct a `FunctionGraph`.

<div class="example" markdown="">

Consider the following example; it constructs two purely symbolic Theano vectors: one with broadcasting and one without.

```{.python}
y_tt = tt.row('y')
print("y_tt.broadcastable = {}".format(y_tt.broadcastable))
x_tt = tt.matrix('x')
print("x_tt.broadcastable = {}".format(x_tt.broadcastable))
```

```{.python}
y_tt.broadcastable = (True, False)
x_tt.broadcastable = (False, False)


```

Notice that it&#x2013;by default&#x2013;signifies no broadcasting on its first and only dimension.

If we wish&#x2013;or if [Theano's configuration demands](http://deeplearning.net/software/theano/library/config.html#config.compute_test_value) it&#x2013;we can assign the symbolic vector arbitrary test values, as long as they're consistent with its type (i.e. a vector, or 1-dimensional array).

In the following, we assign both a broadcastable (i.e. first&#x2013;and only&#x2013;dimension has size 1) and non-broadcastable test value.

Test value is broadcastable:

```{.python}
x_tt.tag.test_value = np.array([[5]])
print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(x_tt.tag.test_value).broadcastable))
print("x_tt.broadcastable = {}".format(x_tt.broadcastable))

# Compute this to run internal checks
try:
    x_tt.shape
    print("shape checks out!")
except TypeError as e:
    print(str(e))
```

```{.python}
test_value.broadcastable = (True, True)
x_tt.broadcastable = (False, False)
shape checks out!


```

```{.python}
y_tt.tag.test_value = np.array([[5]])
print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(y_tt.tag.test_value).broadcastable))
print("y_tt.broadcastable = {}".format(y_tt.broadcastable))

# Compute this to run internal checks
try:
    y_tt.shape
    print("shape checks out!")
except TypeError as e:
    print(str(e))
```

```{.python}
test_value.broadcastable = (True, True)
y_tt.broadcastable = (True, False)
shape checks out!


```

Test value is **not** broadcastable:

```{.python}
x_tt.tag.test_value = np.array([[5, 4]])
print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(x_tt.tag.test_value).broadcastable))
print("x_tt.broadcastable = {}".format(x_tt.broadcastable))

# Compute this to run internal checks
try:
    x_tt.shape
    print("shape checks out!")
except TypeError as e:
    print(str(e))
```

```{.python}
test_value.broadcastable = (True, False)
x_tt.broadcastable = (False, False)
shape checks out!


```

```{.python}
y_tt.tag.test_value = np.array([[5, 4], [3, 2]])
print("test_value.broadcastable = {}".format(
    tt.as_tensor_variable(y_tt.tag.test_value).broadcastable))
print("y_tt.broadcastable = {}".format(y_tt.broadcastable))

# Compute this to run internal checks
try:
    y_tt.shape
    print("shape checks out!")
except TypeError as e:
    print(str(e))
```

```{.python}
test_value.broadcastable = (False, False)
y_tt.broadcastable = (True, False)
For compute_test_value, one input test value does not have the requested type.

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
  File "<ipython-input-163-7eec6ac09cb0>", line 1, in <module>
    __org_babel_python_fname = '/tmp/user/1000/babel-f5w2XO/python-Cjuz3L'; __org_babel_python_fh = open(__org_babel_python_fname); exec(compile(__org_babel_python_fh.read(), __org_babel_python_fname, 'exec')); __org_babel_python_fh.close()
  File "/tmp/user/1000/babel-f5w2XO/python-Cjuz3L", line 1, in <module>
    y_tt = tt.row('y')

The error when converting the test value to that variable type:
Non-unit value on shape on a broadcastable dimension.
(2, 2)
(True, False)


```

Simply put: non-broadcastable Theano tensor variable types can take broadcastable and non-broadcastable values, while broadcastable types can only take broadcastable values.

</div>

What we can take from the example above is that if we determine that a vector has broadcastable dimensions using test values&#x2013;as PyMC3 does&#x2013;we unnecessarily introduce restrictions and potential inconsistencies down the line. One point of origin for such issues is **shared variables**.


# Optimizations Using `RandomVariable`

With our new `RandomVariable`, we can alter the replacement patterns used by `tt.gof.opt.PatternSub` in <sup id="24875a2c31fa7f94ce562adddedc0bf8"><a href="#WillardSymbolicMathPyMC32018" title="@misc{WillardSymbolicMathPyMC32018, title = {Symbolic {{Math}} in {{PyMC3}}}, urldate = {2018-12-27}, url = {https://brandonwillard.github.io/symbolic-math-in-pymc3.html}, author = {Willard, Brandon T.}, month = dec, year = {2018}, file = {/home/bwillard/Zotero/storage/6VVT4UNF/symbolic-math-in-pymc3.html} }">WillardSymbolicMathPyMC32018</a></sup> and implement a slightly better parameter lifting for affine transforms of scalar normal random variables.

```{#rv_optimizations .python}
# Create random variable constructors.
NormalRV = RandomVariable('normal', 0, [0, 0], 'normal')
MvNormalRV = RandomVariable('multivariate_normal', 1, [1, 2], 'multivariate_normal')

# We use the following to handle keyword arguments.
construct_rv = lambda rng, size, mu, sd: NormalRV(mu, sd, size=size, rng=rng)

norm_lift_pats = [
    # Lift element-wise multiplication
    tt.gof.opt.PatternSub(
        (tt.mul,
         'a_x',
         (NormalRV, 'rs_x', 'size_x', 'mu_x', 'sd_x')),
        (construct_rv,
         'rs_x',
         # XXX: Is this really consistent?  How will it handle broadcasting?
         'size_x',
         (tt.mul, 'a_x', 'mu_x'),
         (tt.mul, 'a_x', 'sd_x'),
        )),
    # Lift element-wise addition
    tt.gof.opt.PatternSub(
        (tt.add,
         (NormalRV, 'rs_x', 'size_x', 'mu_x', 'sd_x'),
         'b_x'),
        (construct_rv,
         'rs_x',
         # XXX: Is this really consistent?  How will it handle broadcasting?
         'size_x',
         (tt.add, 'mu_x', 'b_x'),
         'sd_x',
        )),
]

norm_lift_opts = tt.gof.opt.EquilibriumOptimizer(
    norm_lift_pats, max_use_ratio=10)
```

<div class="example" markdown="">

```{#mat_mul_scaling_rv_exa .python}
from theano.gof import FunctionGraph, Feature, NodeFinder
from theano.gof.graph import inputs as tt_inputs, clone_get_equiv

mu_X = tt.vector('mu_X')
sd_X = tt.vector('sd_X')

mu_X.tag.test_value = np.array([0], dtype=tt.config.floatX)
sd_X.tag.test_value = np.array([1, 2], dtype=tt.config.floatX)

# TODO: Defining the offset, `b_tt`, using `tt.vector` will err-out because of
# non-matching dimensions and no default broadcasting.
# This is another good reason for broadcasting inputs in `make_node`.
# E.g.
# b_tt = tt.vector('b')
# b_tt.tag.test_value = np.array([1, 2], dtype=tt.config.floatX)
# or
# b_tt.tag.test_value = np.array([1], dtype=tt.config.floatX)

b_tt = tt.as_tensor_variable([5.], name='b')

X_rv = NormalRV(mu_X, sd_X, name='~X_rv')
Z_rv = 5 * X_rv + b_tt

Z_graph = FunctionGraph(tt_inputs([Z_rv]), [Z_rv])

Z_graph_opt = Z_graph.clone()

_ = norm_lift_opts.optimize(Z_graph_opt)

print('Before: {}'.format(tt.pprint(Z_graph.outputs[0])))
print('After: {}'.format(tt.pprint(Z_graph_opt.outputs[0])))
```

```{.text}
Before: ((TensorConstant{5} * normal_rv(<RandomStateType>, TensorConstant{[]}, mu_X, sd_X)) + TensorConstant{(1,) of 5.0})
After: normal_rv(<RandomStateType>, TensorConstant{[]}, ((TensorConstant{5} * mu_X) + TensorConstant{(1,) of 5.0}), (TensorConstant{5} * sd_X))


```

</div>

Now, what if we wanted to handle affine transformations of a multivariate normal random variable? Specifically, consider implementing the following:

\begin{equation*}
  X \sim N\left(\mu, \Sigma \right), \quad
  A X \sim N\left(A \mu, A \Sigma A^\top \right)
 \;.
\end{equation*}

At first, the following substitution pattern might seem reasonable:

```{.python}
# Vector multiplication
tt.gof.opt.PatternSub(
    (tt.dot,
     'A_x',
     (MvNormalRV, 'rs_x', 'size_x', 'mu_x', 'cov_x')),
    (construct_rv,
     MvNormalRV,
     'rs_x',
     'size_x',
     (tt.dot, 'A_x', 'mu_x'),
     (tt.dot,
      (tt.dot, 'A_x', 'cov_x')
      (tt.transpose, 'A_x')),
    ))
```

Unfortunately, the combination of size parameter and broadcasting complicates the scenario. Both parameters indirectly affect the distribution parameters, making the un-lifted dot-product consistent, but not necessarily the lifted products.

The following example demonstrates the lifting issues brought on by broadcasting.

<div class="example" markdown="">

First, we create a simple multivariate normal.

```{.python}
mu_X = [0, 10]
cov_X = np.diag([1, 1e-2])
size_X_rv = [2, 3]
X_rv = MvNormalRV(mu_X, cov_X, size=size_X_rv)

print('X_rv sample:\n{}\n'.format(X_rv.tag.test_value))
```

```{.python}
X_rv sample:
[[[-0.49226543  9.98771301]
  [-0.22713441 10.00124952]
  [ 1.21604812 10.0386737 ]]

 [[ 1.61758857  9.98456418]
  [ 1.26945358  9.9205853 ]
  [ 1.25295917 10.09953858]]]


```

Next, we create a simple matrix operator to apply to the multivariate normal.

```{.python}
A_tt = tt.as_tensor_variable([[2, 5, 8], [3, 4, 9]])
# or A_tt = tt.as_tensor_variable([[2, 5, 8]])

# It's really just `mu_X`...
E_X_rv = X_rv.owner.inputs[2]

print('A * X_rv =\n{}\n'.format(tt.dot(A_tt, X_rv).tag.test_value))
```

```{.python}
A * X_rv =
[[[  7.60818207 150.2910632 ]
  [ 19.60611837 150.36836345]]

 [[  8.55909917 160.31620039]
  [ 21.20721252 160.53188091]]]


```

As we can see, the multivariate normal's test/sampled value has the correct shape for our matrix operator.

```{.python}
import traceback
try:
    print('A * E[X_rv] =\n{}\n'.format(tt.dot(A_tt, E_X_rv).tag.test_value))
except ValueError as e:
    print("".join(traceback.format_exception_only(type(e), e)))
```

```{.python}
ValueError: shapes (2,3) and (2,) not aligned: 3 (dim 1) != 2 (dim 0)


```

However, we see that the multivariate normal's inputs (i.e. the `Op` inputs)&#x2013;specifically the mean parameter&#x2013;do not directly reflect the support's shape, as intuitively would suggest.

```{.python}
size_tile = tuple(size_X_rv) + (1,)
E_X_rv_ = tt.tile(E_X_rv, size_tile, X_rv.ndim)

print('A * E[X_rv] =\n{}\n'.format(tt.dot(A_tt, E_X_rv_).tag.test_value))
```

```{.python}
A * E[X_rv] =
[[[  0 150]
  [  0 150]]

 [[  0 160]
  [  0 160]]]


```

We can manually replicate the inputs so that they match the output shape, but a solution to the general problem requires a more organized response.

</div>


# Discussion

In a follow-up, we'll address a few loose ends, such as

-   the inclusion of density functions and likelihoods,
-   decompositions/reductions of overlapping multivariate types (e.g. transforms between tensors of univariate normals and equivalent multivariate normals),
-   canonicalization of graphs containing `RandomVariable` terms,
-   and optimizations that specifically benefit MCMC schemes (e.g. automatic conversion to scale mixture decompositions that improve sampling/covariance structure).

# Bibliography
<a id="WillardSymbolicMathPyMC32018"></a>[WillardSymbolicMathPyMC32018] Willard, Symbolic Math in PyMC3, <i></i>, (2018). <a href="https://brandonwillard.github.io/symbolic-math-in-pymc3.html">link</a>. [↩](#24875a2c31fa7f94ce562adddedc0bf8)
