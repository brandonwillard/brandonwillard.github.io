---
bibliography:
- '/home/bwillard/projects/websites/brandonwillard.github.io/content/articles/src/tex/dlm-optimizations.bib'
modified: '2020-4-23'
status: draft
tags: 'draft,pymc3,theano,statistics,symbolic computation,python,probability theory'
title: Dynamic Linear Model Optimizations in Theano
date: '2020-03-18'
author: 'Brandon T. Willard'
figure_dir: '{attach}/articles/figures/'
figure_ext: png
---

<div class="abstract">
In this document we construct dynamic linear models (DLMs) in Theano and explore ideas for automating the production of efficient samplers.

</div>


# Introduction

We start by considering the simple form of a DLM <a id="4bbd465b4e78e5c5151b0cbba54d984e"><a href="#harrison_bayesian_1999">(Harrison &amp; West 1999)</a></a> with prior \(\theta_0 \sim \operatorname{N}\left( m_0, C_0 \right)\):

\begin{align}
  y_t &= F_t^{\top} \theta_{t} + \epsilon_t, \quad \epsilon_t \sim \operatorname{N}\left( 0, V \right)
  \label{eq:basic-dlm-obs}
  \\
  \theta_t &= G_t \theta_{t-1} + \nu_t, \quad \nu_t \sim \operatorname{N}\left( 0, W \right)
  \label{eq:basic-dlm-state}
\end{align}

for \(t \in \{1, \dots, T\}\), \(y_t \in \mathbb{R}\), and \(\theta_t \in \mathbb{R}^{M}\).

The most "notationally" faithful representation of the timeseries model in \(\eqref{eq:basic-dlm-state}\) using Theano is provided in Listing [2](#org27bfd8a). It represents the notion of a recursion&#x2013;to the best of Theano's ability&#x2013;by way of the `scan` operator.

<figure>
```{.python}
import numpy as np

import theano
import theano.tensor as tt

import matplotlib.pyplot as plt

plt.style.use('ggplot')
plt_orig_cycler = plt.rcParams['axes.prop_cycle']
plt.rc('text', usetex=True)

from theano.printing import debugprint as tt_dprint

from symbolic_pymc.theano.random_variables import NormalRV, MvNormalRV, GammaRV, observed


# theano.config.cxx = ""
# theano.config.mode = "FAST_COMPILE"
tt.config.compute_test_value = 'ignore'
```
</figure>

<figure id="org27bfd8a">
```{.python}

N_obs_tt = tt.iscalar("N_obs")
N_theta_tt = tt.iscalar("N_theta")

G_tt = tt.specify_shape(tt.matrix(), [N_theta_tt, N_theta_tt])
G_tt.name = 'G_t'

F_tt = tt.specify_shape(tt.col(), [N_theta_tt, 1])
F_tt.name = 'F_t'

rng_state = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(1234)))
rng_init_state = rng_state.get_state()
rng_tt = theano.shared(rng_state, name='rng', borrow=True)
rng_tt.tag.is_rng = True
rng_tt.default_update = rng_tt

m_0_tt = tt.zeros([N_theta_tt])
m_0_tt.name = "m_0"
C_0_tt = 10. * tt.eye(N_theta_tt)
C_0_tt.name = "C_0"

theta_0_rv = MvNormalRV(m_0_tt, C_0_tt, rng=rng_tt, name='theta_0')

nu_scale_tt = theano.shared(np.r_[1.1, 10.0], name='nu_scale')
W_tt = tt.eye(N_theta_tt) * tt.inv(nu_scale_tt)
W_tt.name = "W_t"

eps_scale_tt = theano.shared(0.7, name='eps_scale')
V_tt = tt.inv(eps_scale_tt)
V_tt.name = "V_t"

def state_step(theta_tm1, G_t, W_t, N_theta, rng):
    nu_rv = MvNormalRV(tt.zeros([N_theta]),
                       W_t,
                       rng=rng,
                       name='nu')
    theta_t = G_t.dot(theta_tm1) + nu_rv
    return theta_t


theta_t_rv, theta_t_updates = theano.scan(fn=state_step,
                                          outputs_info={"initial": theta_0_rv, "taps": [-1]},
                                          non_sequences=[G_tt, W_tt, N_theta_tt, rng_tt],
                                          n_steps=N_obs_tt,
                                          #strict=True,
                                          name='theta_t')

def obs_step(theta_t, F_t, V_t, rng):
    eps_rv = NormalRV(0.0, V_t,
                      rng=rng,
                      name='eps')
    y_t = F_t.T.dot(theta_t) + eps_rv
    return y_t


Y_t_rv, Y_t_updates = theano.scan(fn=obs_step,
                                  sequences=[theta_t_rv],
                                  non_sequences=[F_tt, V_tt, rng_tt],
                                  #strict=True,
                                  name='Y_t')
```
<figcaption>Listing 2</figcaption>
</figure>

The model in Listing [2](#org27bfd8a) is our starting point. We assume that a PyMC3 user&#x2013;for instance&#x2013;would define a timeseries model in this way, alongside distributional assumptions on parameters (e.g. inverse-gamma variances). From there, we'll explore some ideas behind manually producing model-specific efficient samplers&#x2013;generally by first manually deriving and then demonstrating said samplers.

Throughout we'll use data sampled from \(\eqref{eq:basic-dlm-state}\) for demonstration purposes. Specifically, our simulation has the following values:

\begin{gather}
  T = 200,\quad M = 2
  \\
  G_t = \begin{pmatrix}
  1 & 0.1 \\
  0 & 1 \\
  \end{pmatrix},\quad
  F_t = \begin{pmatrix}
  1 \\
  0.2
  \end{pmatrix}
  \\
  \theta_0 = \begin{pmatrix}
  0 \\
  0
  \end{pmatrix}
  \label{eq:sim-settings}
\end{gather}

<figure>
```{.python}
from theano import function as tt_function

dlm_sim_values = {
    N_obs_tt: 200,
    N_theta_tt: 2,
    G_tt: np.r_['0,2',
                [1.0, 0.1],
                [0.0, 1.0]].astype(tt.config.floatX),
    F_tt: np.r_[[[1.0],
                 [0.2]]].astype(tt.config.floatX)
}

rng_tt.get_value(borrow=True).set_state(rng_init_state)

simulate_dlm = tt_function([N_obs_tt, N_theta_tt, G_tt, F_tt],
                           [Y_t_rv, theta_t_rv],
                           givens={theta_0_rv: np.r_[0.0, 0.0]},
                           updates=Y_t_updates)

y_sim, theta_t_sim = simulate_dlm(dlm_sim_values[N_obs_tt], dlm_sim_values[N_theta_tt], dlm_sim_values[G_tt], dlm_sim_values[F_tt])

# rng_sim_state = rng_tt.get_value(borrow=True).get_state()
```
</figure>

In [4](#orgece0fee) we plot a sample from the model in Listing [2](#org27bfd8a) for a fixed RNG seed.

<figure id="orgece0fee">
```{.python}
plt.clf()
_ = plt.plot(y_sim, label=r'$y_t$', color='black', linewidth=0.7)
plt.tight_layout()
plt.legend()
```
<figcaption>Listing 4</figcaption>
</figure>

<figure id="nil" class="plot"> ![ \label{nil}]({attach}/articles/figures/basic-dlm-sim-plot.png) <figcaption></figcaption> </figure>

Since our goal is to automate some of the basic steps in the process of analytically manipulating and/or solving DLMs (for the purpose of producing efficient and accurate posterior estimates), we will want to compute as many closed-form operations as possible, and the prior predictive state and observation distributions are a good place to start.

Given all the prior and observed data up to time \(t\), \(D_t\), these distribution are given by the following:

\begin{align}
  \theta_{t} \mid D_{t-1} &\sim \operatorname{N}\left( a_{t}, R_{t} \right)
  \\
  y_{t} \mid D_{t-1} &\sim \operatorname{N}\left( f_{t}, Q_{t} \right)
\end{align}

The prior predictive moments are as follows:

\begin{equation}
  \begin{gathered}
    a_t = G_t m_{t-1}, \quad R_t = G_t C_{t-1} G_t^\top + W_t
    \\
    f_t = F_t^\top a_{t}, \quad Q_t = F_t^\top C_{t-1} F_t + V_t
  \end{gathered}
  \label{eq:dlm-prior-predictive}
\end{equation}

We'll also want to compute the posterior moments for \(\theta_t \mid D_t\), which are as follows:

\begin{equation}
  \begin{gathered}
    m_t = a_{t} + R_t F_t Q_t^{-1} \left(y_t - f_t\right),
    \quad C_t = R_t  - R_t F_t Q_t^{-1} F_t^\top R_t
  \end{gathered}
  \label{eq:dlm-post-moments}
\end{equation}

These "filtered" moments/distributions are only **one** kind of posterior result for a DLM, and they only take into account the data up to time \(t\). The other kind are the "smoothed" distributions, which provided posterior distributions for each time \(t\) given all observations.

Notationally, we've used \(D_t\) to signify all conditional observations and parameters up to time \(t\), so the smoothed distributions are given by \(\theta_t \mid D_T\) and the following moments:

\begin{equation}
  \begin{aligned}
    s_t &= m_t + C_t G_{t+1}^\top R_{t+1}^{-1} \left( s_{t+1} - a_{t+1} \right)
    \\
    S_t &= C_t - C_t G_{t+1}^\top R_{t+1}^{-1} \left( R_{t+1} - S_{t+1} \right) R_{t+1}^{-1} G_{t+1} C_t
  \end{aligned}
  \label{eq:dlm-smooth-moments}
\end{equation}

<div class="remark" markdown="">
In most cases, models will not be as simple as the standard DLM. Even so, these basic closed-form solutions can still be relevant. For instance, efficient MCMC algorithms can be constructed using these closed-form results for **conditionally linear** models. In those cases, we can compute the posterior moments&#x2013;in closed-form&#x2013;conditional on samples generated by other means.

</div>

The standard approach is called forward-filtering backward-sampling (FFBS) and uses smoothed posteriors \(\theta_t \mid \theta_{t+1}, D_T\) conditioned on all other parameters.

We'll build up to forward-backward sampling in what follows, but, first, we need to establish how the requisite quantities can be computed symbolically.


# Posterior Estimation

In Listings [6](#org7278e18) and [7](#org874cba4), we demonstrate how the posterior moments in \(\eqref{eq:dlm-post-moments}\) and \(\eqref{eq:dlm-smooth-moments}\) can be computed in Theano.

Unfortunately, if we attempt to implement the exact closed-form updates in \(\eqref{eq:dlm-post-moments}\) or \(\eqref{eq:dlm-smooth-moments}\), our results will be fraught with numerical errors. This is a very basic issue with naively implemented Kalman filters. The solution to these issues usually involves some analytic reformulations that compensate for the covariance matrix subtractions. The standard approaches generally use some form of matrix decomposition that directly accounts for the positive semi-definite nature of the covariance matrices.

The approach taken here is based on the singular value decomposition (SVD) and effectively computes only one symmetric "half" of the updated covariances. The SVD also allows for easy inversions. See <a id="0ae04c048b20d07f32d7f0f75bb51483"><a href="#ZhangFixedintervalsmoothingalgorithm1996">Zhang &amp; Li (1996)</a></a> for more details, or <a id="3a4d89388a434d7b1b91dc8690f3a03b"><a href="#PetrisDynamiclinearmodels2009">Petris, Petrone &amp; Campagnoli (2009)</a></a> for a concise overview of the procedure in the context of DLMs.

<figure>
```{.python}
import scipy
import warnings

from theano.gof import Op, Apply
from theano.tensor.opt import Assert
from theano.tensor.slinalg import Solve, MATRIX_STRUCTURES
from theano.tensor.nlinalg import matrix_dot

warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-tuple sequence")


def tt_finite_inv(x):
    y = tt.inv(x)
    res_subtensor = y[tt.isinf(y)]
    return tt.set_subtensor(res_subtensor, 0.0)

```
</figure>


## SVD Approach

TODO: Describe SVD filter formulation.

<figure id="org7278e18">
```{.python}
from theano.tensor.nlinalg import svd


y_tt = tt.specify_shape(tt.col(), [N_obs_tt, 1])
y_tt.name = 'y_t'


def filtering_step(y_t, m_tm1, U_C_tm1, S_C_tm1, F_t, G_t, N_W_t, U_V_t, S_V_inv_t):
    """Compute the sequential posterior state and prior predictive parameters."""

    M_R = tt.join(0,
                  matrix_dot(S_C_tm1, U_C_tm1.T, G_t.T),
                  N_W_t)
    # TODO: Consider an approach that only computes *one* set of singular
    # vectors
    _, d_M_R, Vt_M_R = svd(M_R)
    Vt_M_R.name = "Vt_M_R"

    U_R_t, s_R_t = Vt_M_R.T, d_M_R
    U_R_t.name = "U_R_t"

    # R_t = M_R.T.dot(M_R) = matrix_dot(U_R_t, tt.diag(d_M_R), U_R_t.T)

    # V_t_inv = N_V_t_inv.T @ N_V_t_inv
    N_V_t_inv = S_V_inv_t.dot(U_V_t.T)
    N_V_t_inv.name = "N_V_t_inv"

    M_C = tt.join(0,
                  matrix_dot(N_V_t_inv, F_t.T, U_R_t),
                  tt.diag(tt_finite_inv(s_R_t)))
    # TODO: Consider an approach that only computes *one* set of singular
    # vectors
    _, d_M_C, Vt_M_C = svd(M_C)
    Vt_M_C.name = "Vt_M_C"

    U_C_t, D_C_t = U_R_t.dot(Vt_M_C.T), tt.diag(tt_finite_inv(d_M_C))
    U_C_t.name = "U_C_t"
    D_C_t.name = "D_C_t"

    C_t = matrix_dot(U_C_t, D_C_t, U_C_t.T)
    C_t.name = "C_t"

    a_t = G_t.dot(m_tm1)
    a_t.name = "a_t"
    f_t = F_t.T.dot(a_t)
    f_t.name = "f_t"
    m_t = a_t + matrix_dot(C_t, F_t, N_V_t_inv.T, N_V_t_inv, y_t - f_t)
    m_t.name = "m_t"

    S_C_t = tt.sqrt(D_C_t)
    S_C_t.name = "S_C_t"

    S_R_t = tt.diag(s_R_t)
    S_R_t.name = "S_R_t"

    return [m_t, U_C_t, S_C_t, a_t, U_R_t, S_R_t]


U_C_0_tt, d_C_0_tt, _ = svd(C_0_tt)
S_C_0_tt = tt.diag(tt.sqrt(d_C_0_tt))
S_C_0_tt.name = "S_C_0_tt"

U_W_tt, d_W_tt, _ = svd(W_tt)
s_W_tt = tt.sqrt(d_W_tt)
N_W_tt = tt.diag(s_W_tt).dot(U_W_tt.T)
N_W_tt.name = "N_W"

U_V_tt, D_V_tt, _ = svd(tt.as_tensor_variable(V_tt, ndim=2) if V_tt.ndim < 2 else V_tt)
S_V_inv_tt = tt.diag(tt_finite_inv(tt.sqrt(D_V_tt)))
# N_V_tt = S_V_tt.dot(U_V_tt.T)

(m_t, U_C_t, S_C_t, a_t, U_R_t, S_R_t), filter_updates = theano.scan(fn=filtering_step,
                                                   sequences=y_tt,
                                                   outputs_info=[
                                                       {"initial": m_0_tt, "taps": [-1]},
                                                       {"initial": U_C_0_tt, "taps": [-1]},
                                                       {"initial": S_C_0_tt, "taps": [-1]},
                                                       {}, {}, {}  # a_t, U_R_t, S_R_t
                                                   ],
                                                   non_sequences=[F_tt, G_tt, N_W_tt, U_V_tt, S_V_inv_tt],
                                                   strict=True,
                                                   name='theta_t_obs')
```
<figcaption>Listing 6</figcaption>
</figure>

TODO: Describe special manipulations behind SVD smoother.

<figure id="org874cba4">
```{.python}

def smoother_step(m_t, U_C_t, S_C_t, a_tp1, U_R_tp1, S_R_tp1, m_Ttp1, U_C_Ttp1, S_C_Ttp1, G_tp1, N_W_t_inv):
    """Smooth a series starting from the "forward"/sequentially computed posterior moments."""

    N_C_t = S_C_t.dot(U_C_t.T)

    S_R_tp1_inv = tt_finite_inv(S_R_tp1)
    N_R_tp1_inv = S_R_tp1_inv.dot(U_R_tp1.T)

    # B_t = C_t @ G_tp1.T @ R_tp1
    B_t = matrix_dot(N_C_t.T, N_C_t, G_tp1.T, N_R_tp1_inv.T, N_R_tp1_inv)

    S_C_t_inv = tt_finite_inv(S_C_t)

    # M_H_t.T @ M_H_t = G_tp1 @ W_t_inv @ G_tp1.T + C_t_inv
    M_H_t_inv = tt.join(0,
                        N_W_t_inv.dot(G_tp1),
                        S_C_t_inv.dot(U_C_t.T))
    _, d_H_t_inv, U_H_t = svd(M_H_t_inv)

    # H_t = inv(M_H_t.T @ M_H_t) = C_t - B_t @ R_tp1 @ B_t.T
    D_H_t = tt.diag(tt_finite_inv(d_H_t_inv))

    # C_Tt = C_t - matrix_dot(B_t, R_tp1 - C_Ttp1, B_t.T)
    # C_Tt = M_C_Ttp1.T.dot(M_C_Ttp1)
    M_C_Tt = tt.join(0,
                     D_H_t.dot(U_H_t),
                     matrix_dot(S_C_Ttp1, U_C_Ttp1.T, B_t.T))
    U_C_Tt, d_C_Tt, _ = svd(M_C_Tt)

    S_C_Tt = tt.diag(tt.sqrt(d_C_Tt))

    m_Tt = m_t + B_t.dot(m_Ttp1 - a_tp1)

    return [m_Tt, U_C_Tt, S_C_Tt]


N_W_inv_tt = tt.diag(tt_finite_inv(s_W_tt)).dot(U_W_tt.T)

m_T = m_t[-1]
U_C_T = U_C_t[-1]
S_C_T = S_C_t[-1]

# These series only go from N_obs - 1 to 1
(m_Tt_rev, U_C_Tt_rev, S_C_Tt_rev), _ = theano.scan(fn=smoother_step,
                                                    sequences=[
                                                        {"input": m_t, "taps": [-1]},
                                                        {"input": U_C_t, "taps": [-1]},
                                                        {"input": S_C_t, "taps": [-1]},
                                                        {"input": a_t, "taps": [1]},
                                                        {"input": U_R_t, "taps": [1]},
                                                        {"input": S_R_t, "taps": [1]}
                                                    ],
                                                    outputs_info=[
                                                        {"initial": m_T, "taps": [-1]},
                                                        {"initial": U_C_T, "taps": [-1]},
                                                        {"initial": S_C_T, "taps": [-1]},
                                                    ],
                                                    non_sequences=[G_tt, N_W_inv_tt],
                                                    go_backwards=True,
                                                    strict=True,
                                                    name='theta_Tt_obs')

m_Tt = m_Tt_rev[::-1]
U_C_Tt = U_C_Tt_rev[::-1]
S_C_Tt = S_C_Tt_rev[::-1]

m_Tt = tt.join(0, m_Tt, [m_T])
U_C_Tt = tt.join(0, U_C_Tt, [U_C_T])
S_C_Tt = tt.join(0, S_C_Tt, [S_C_T])
```
<figcaption>Listing 7</figcaption>
</figure>

Listing [8](#orgad4c1a3) computes the filtered and smoothed means for our simulated series, and Figure [9](#org50d26f8) shows the results.

<figure id="orgad4c1a3">
```{.python}
filter_smooth_dlm = tt_function([y_tt, N_theta_tt, G_tt, F_tt],
                                [m_t, m_Tt],
                                # mode=theano.compile.mode.FAST_COMPILE
                                )

m_t_sim, m_Tt_sim = filter_smooth_dlm(y_sim, dlm_sim_values[N_theta_tt], dlm_sim_values[G_tt], dlm_sim_values[F_tt])
```
<figcaption>Listing 8</figcaption>
</figure>

<figure id="org50d26f8">
```{.python}
from cycler import cycler

bivariate_cycler = plt_orig_cycler * cycler('linestyle', ['-', '--'])
plt.close(fig='all')

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.set_prop_cycle(bivariate_cycler)
ax.plot(theta_t_sim, label=r'$\theta_t$', linewidth=0.8)
ax.plot(m_t_sim, label=r'$E[\theta_t \mid D_{t}]$', alpha=0.7, linewidth=0.8)
ax.plot(m_Tt_sim, label=r'$E[\theta_t \mid D_{T}]$', alpha=0.7, linewidth=0.8)
plt.legend(framealpha=0.4)
plt.tight_layout()
```
<figcaption>Listing 9</figcaption>
</figure>

<figure id="nil" class="plot"> ![ \label{nil}]({attach}/articles/figures/svd-steps-sim-plot.png) <figcaption></figcaption> </figure>


# Forward-backward Estimation

We can use the smoothing and filtering steps in the previous section to perform a more efficient MCMC estimation than would otherwise be possible without the implicit Rao-Blackwellization.

Forward-filtering backward-sampling <a id="66fb99775f308e808a193bd7bb2d2038"><a href="#Fruhwirth-SchnatterDataaugmentationdynamic1994">(Fr\uhwirth-Schnatter 1994)</a></a> works by first computing the forward filtered moments, allowing one to draw \(\theta_T\) from \( \left(\theta_T \mid D_T\right) \sim \operatorname{N}\left(m_T, C_T\right) \) and, subsequently, \(\theta_t\) from \(\left(\theta_t \mid \theta_{t+1}, D_T \right) \sim \operatorname{N}\left(h_t, H_t\right)\).

The latter distribution's moments are essentially a result of smoothing:

\begin{gather}
  B_t = C_t G^\top_{t+1} R_{t+1}^{-1}
  \\
  h_t = m_t + B_t \left(\theta_{t+1} - a_{t+1}\right)
  \\
  H_t = C_t - B_t R_{t+1} B^\top_t
\end{gather}

TODO: Describe SVD formulation.

<figure>
```{.python}
def ffbs_step(m_t, U_C_t, S_C_t, a_tp1, U_R_tp1, S_R_tp1, theta_tp1, F_tp1, G_tp1, N_W_t_inv, rng):
    """Perform forward-filtering backward-sampling."""

    S_C_t_inv = tt_finite_inv(S_C_t)

    # M_H_t.T @ M_H_t = G_tp1 @ W_t_inv @ G_tp1.T + C_t_inv
    M_H_t_inv = tt.join(0,
                        N_W_t_inv.dot(G_tp1),
                        S_C_t_inv.dot(U_C_t.T))
    _, d_H_t_inv, U_H_t = svd(M_H_t_inv)

    # H_t = inv(M_H_t.T @ M_H_t) = C_t - B_t @ R_tp1 @ B_t.T
    D_H_t = tt.diag(tt_finite_inv(d_H_t_inv))

    # H_t = matrix_dot(U_H_t, D_H_t, U_H_t.T)
    # H_t.name = "H_t"

    N_C_t = S_C_t.dot(U_C_t.T)

    S_R_tp1_inv = tt_finite_inv(S_R_tp1)
    N_R_tp1_inv = S_R_tp1_inv.dot(U_R_tp1.T)

    # B_t = C_t @ G_tp1.T @ R_tp1
    B_t = matrix_dot(N_C_t.T, N_C_t, G_tp1.T, N_R_tp1_inv.T, N_R_tp1_inv)

    h_t = m_t + B_t.dot(theta_tp1 - a_tp1)
    h_t.name = 'h_t'

    # theta_t = MvNormalRV(h_t, H_t, rng=rng, name='theta_t_ffbs')
    theta_t = h_t + matrix_dot(U_H_t, tt.sqrt(D_H_t),
                               MvNormalRV(tt.zeros_like(h_t),
                                          tt.eye(h_t.shape[0]),
                                          rng=rng)
                               )

    # These are statistics we're gathering for other posterior updates
    theta_tp1_diff = theta_tp1 - G_tp1.dot(theta_t)
    f_tp1 = F_tp1.T.dot(theta_t)

    # Sequentially sample/update quantities conditional on `theta_t` here...

    return [theta_t, theta_tp1_diff, f_tp1]


C_T = matrix_dot(U_C_T, tt.square(S_C_T), U_C_T.T)
theta_T_post = MvNormalRV(m_T, C_T, rng=rng_tt)
theta_T_post.name = "theta_T_post"

ffbs_output, ffbs_updates = theano.scan(fn=ffbs_step,
                                        sequences=[
                                            {"input": m_t, "taps": [-1]},
                                            {"input": U_C_t, "taps": [-1]},
                                            {"input": S_C_t, "taps": [-1]},
                                            {"input": a_t, "taps": [1]},
                                            {"input": U_R_t, "taps": [1]},
                                            {"input": S_R_t, "taps": [1]}
                                        ],
                                        outputs_info=[
                                            {"initial": theta_T_post, "taps": [-1]},
                                            {}, {}, # theta_tp1_diff, f_tp1
                                        ],
                                        non_sequences=[F_tt, G_tt, N_W_inv_tt, rng_tt],
                                        go_backwards=True,
                                        strict=True,
                                        name='ffbs_samples')

(theta_t_post_rev, theta_t_diff_rev, f_t_rev) = ffbs_output

theta_t_post = tt.join(0, theta_t_post_rev[::-1], [theta_T_post])

# We need to add the missing end-points onto these statistics...
f_t_post = tt.join(0, f_t_rev[::-1], [F_tt.T.dot(theta_T_post)])

theta_t_diff_rev = tt.join(0, theta_t_diff_rev, [theta_t_post[-1] - G_tt.dot(theta_0_rv)])
```
</figure>

<figure>
```{.python}
# E[nu[0]] = 2.0, Var[nu[0]] = 10.0
# E[nu[1]] = 5.0, Var[nu[1]] = 5.0
a_nu, b_nu = np.r_[2.0**2 / 10.0, 5.0**2 / 5.0], np.r_[2.0 / 10.0, 5.00 / 5.0]

a_eps, b_eps = 0.5, 1.0

nu_post_tt = GammaRV(a_nu + N_obs_tt * 0.5,
                     b_nu + 0.5 * tt.square(theta_t_diff_rev).sum(0),
                     rng=rng_tt, name='nu_post')

eps_post_tt = GammaRV(a_eps + N_obs_tt * 0.5,
                      b_eps + 0.5 * tt.square(y_tt - f_t_post).sum(),
                      rng=rng_tt, name='eps_post')
```
</figure>

<figure>
```{.python}
ffbs_dlm = tt_function([y_tt, N_obs_tt, N_theta_tt, G_tt, F_tt],
                       [theta_t_post, nu_post_tt, eps_post_tt],
                       updates=ffbs_updates)

nu_scale_tt.set_value(np.random.gamma(a_nu, scale=1.0/b_nu))
eps_scale_tt.set_value(np.random.gamma(a_eps, scale=1.0/b_eps))

chain = 0
posterior_samples = {'theta': [[]], 'nu': [[]], 'eps': [[]]}

for i in range(1000):

    theta_t_post_sim, nu_post_sim, eps_post_sim, nu_a, nu_b, eps_a, eps_b = ffbs_dlm(
        y_sim, dlm_sim_values[N_obs_tt], dlm_sim_values[N_theta_tt], dlm_sim_values[G_tt], dlm_sim_values[F_tt])

    # Update variance scale parameters
    nu_scale_tt.set_value(nu_post_sim)
    eps_scale_tt.set_value(eps_post_sim)

    posterior_samples['theta'][chain].append(theta_t_post_sim)
    posterior_samples['nu'][chain].append(nu_post_sim)
    posterior_samples['eps'][chain].append(eps_post_sim)

    print(f'i={i},\tnu={nu_post_sim},\teps={eps_post_sim}')

posterior_samples = {k: np.asarray(v) for k,v in posterior_samples.items()}
```
</figure>

<figure>
```{.python}
from cycler import cycler
from matplotlib.collections import LineCollection


plt.clf()

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.autoscale(enable=False)

# bivariate_cycler =  cycler('linestyle', ['-', '--']) * plt_orig_cycler
# ax.set_prop_cycle(bivariate_cycler)

thetas_shape = posterior_samples['theta'][0].shape

cycle = ax._get_lines.prop_cycler

for d in range(thetas_shape[-1]):

    styles = next(cycle)
    thetas = posterior_samples['theta'][0].T[d].T

    theta_lines = np.empty(thetas_shape[:-1] + (2,))
    theta_lines.T[0] = np.tile(np.arange(thetas_shape[-2]), [thetas_shape[-3], 1]).T
    theta_lines.T[1] = thetas.T

    ax.add_collection(
        LineCollection(theta_lines,
                       label=r'$\theta_t \mid D_{T}$',
                       alpha=0.3, linewidth=0.9,
                       **styles)
    )

bivariate_obs_cycler =  cycler('linestyle', ['-', '--']) * cycler('color', ['black'])

ax.set_prop_cycle(bivariate_obs_cycler)
ax.plot(theta_t_sim, label=r'$\theta_t$', linewidth=1.0)

ax.autoscale(enable=True)

plt.tight_layout()

plt.legend(framealpha=0.4)
```
</figure>

<figure id="nil" class="plot"> ![ \label{nil}]({attach}/articles/figures/ffbs-sim-plot.png) <figcaption></figcaption> </figure>

<figure>
```{.python}
import arviz as az

az_trace = az.from_dict(posterior=posterior_samples)
az.plot_trace(az_trace, compact=True)
```
</figure>

<figure id="nil" class="plot"> ![ \label{nil}]({attach}/articles/figures/ffbs-trace-plot.png) <figcaption></figcaption> </figure>


# Discussion

So far, we've only shown how to perform FFBS for DLMs in Theano.

# Bibliography
<a id="harrison_bayesian_1999"></a> Harrison & West, Bayesian Forecasting & Dynamic Models, Springer (1999). [↩](#4bbd465b4e78e5c5151b0cbba54d984e)

<a id="ZhangFixedintervalsmoothingalgorithm1996"></a> Zhang & Li, Fixed-Interval Smoothing Algorithm Based on Singular Value Decomposition, 916-921, in in: , Proceedings of the 1996 IEEE International Conference on Control Applications, 1996, edited by (1996) [↩](#0ae04c048b20d07f32d7f0f75bb51483)

<a id="PetrisDynamiclinearmodels2009"></a> Petris, Petrone & Campagnoli, Dynamic Linear Models, Springer (2009). [↩](#3a4d89388a434d7b1b91dc8690f3a03b)

<a id="Fruhwirth-SchnatterDataaugmentationdynamic1994"></a> Fr\"uhwirth-Schnatter, Data Augmentation and Dynamic Linear Models, <i>Journal of time series analysis</i>, <b>15(2)</b>, 183-202 (1994). <a href="http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9892.1994.tb00184.x/abstract">link</a>. [↩](#66fb99775f308e808a193bd7bb2d2038)
