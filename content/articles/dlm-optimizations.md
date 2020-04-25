---
bibliography:
- '/home/bwillard/projects/websites/brandonwillard.github.io/content/articles/src/tex/dlm-optimizations.bib'
modified: '2020-4-24'
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

phi_W_tt = theano.shared(np.r_[1.1, 10.0], name='phi_W')
W_tt = tt.eye(N_theta_tt) * tt.inv(phi_W_tt)
W_tt.name = "W_t"

phi_V_tt = theano.shared(0.7, name='phi_V')
V_tt = tt.inv(phi_V_tt)
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
                                          strict=True,
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
                                  strict=True,
                                  name='Y_t')
```
<figcaption>Listing 2</figcaption>
</figure>

The model in Listing [2](#org27bfd8a) is our starting point. We assume that a PyMC3 user&#x2013;for instance&#x2013;would define a timeseries model in this way, alongside distributional assumptions on parameters (e.g. inverse-gamma variances). From there, we'll explore some ideas behind manually producing model-specific efficient samplers&#x2013;generally by first manually deriving and then demonstrating said samplers.

Throughout we'll use data sampled from \(\eqref{eq:basic-dlm-state}\) for demonstration purposes. Specifically, our simulation has the following values:

\begin{gather}
  T = 200,\quad M = 2
  \\
  \phi_W = \left(1.1, 10\right),\quad \phi_V = 0.7
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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message="Using a non-tuple sequence")


def tt_finite_inv(x):
    y = tt.inv(x)
    res_subtensor = y[tt.isinf(y)]
    return tt.set_subtensor(res_subtensor, 0.0)

```
</figure>


## SVD-based Filtering

The SVD forms of the filtering equations in \(\eqref{eq:dlm-post-moments}\) are produced through creative use of the SVDs of its component matrices. Using a slightly modified version of the formulation established in <a id="3a4d89388a434d7b1b91dc8690f3a03b"><a href="#PetrisDynamiclinearmodels2009">Petris, Petrone &amp; Campagnoli (2009)</a></a>, the SVD for a matrix \(M\) is given by \(M = U_{M} D_{M} V_{M}^\top\). A symmetric matrix then takes the form \(M = U_{M} D_{M} U_{M}^\top\) and its "square-root" is given by \(M = N_M^\top N_M\) with \(N_M = S_{M} U_{M}^\top\) and \(S_{M} = D_{M}^{1/2}\). Likewise, matrix (generalized) inverses take the form \(M^{-1} = U_{M} S_{M}^{-1} U_{M}^\top\).

The idea here is that we can combine these SVD identities to derive square-root relationship between the SVD of \(C_t^{-1}\) and the SVDs of \(C_{t-1}\), \(W_t\), \(V_t\), and \(R_t\), then we can easily invert \(C_t^{-1}\) to arrive at the desired numerically stable SVD of \(C_t\).

First, note that \(N_{R_t}^\top N_{R_t} = G_t C_{t-1} G_t^\top + W_t = R_t\) for

\begin{equation}
  \begin{aligned}
    N_{R_t} &=
      \begin{pmatrix}
        S_{C_{t-1}} U_{C_{t-1}} G_t^\top
        \\
        N_{W_t}
      \end{pmatrix}
  \end{aligned}
  .
  \label{eq:N_R_t}
\end{equation}

From this, we know that the (symmetric) SVD of \(R_t\) is equal to the (symmetric) SVD of \(N_{R_t}\), i.e. \(U_{R_t} = V_{N_{R_t}}\) and \(S_{R_t} = D_{N_{R_t}}\). In other words, we have to compute the SVD of \(N_{R_t}\) at this step.

With the updated SVD of \(R_t\), we can use \(C_t^{-1} = F_t V_t^{-1} F_t^\top + R_t^{-1}\)&#x2013;obtained via the classic [Sherman-Morrison-Woodbury matrix inverse identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)&#x2013;to employ the same technique as before and produce the SVD of \(C_t^{-1}\) by way of the SVD of yet another block square-root matrix,

\begin{equation}
  \begin{aligned}
    N_{C_t}^{-1} &=
      \begin{pmatrix}
        N_{V_t} F_t^\top U_{R_t}
        \\
        S_{R_t}^{-1}
      \end{pmatrix}
  \end{aligned}
  .
  \label{eq:N_C_t_inv}
\end{equation}

Again, we compute the SVD of \(N_{C_t^{-1}}\) at this step to obtain \(V_{N_{C_t}^{-1}}\) and \(D_{N_{C_t}^{-1}}\).

This time, the block square-root matrix relationship isn't direct: \(U_{R_t} N_{C_t}^{-\top} N_{C_t}^{-1} U_{R_t}^\top = C_t^{-1}\). However, since the extra \(U_{R_t}\) terms are orthogonal, we are able to derive the SVD of \(C_t\) as \(U_{C_t} = U_{R_t} V_{N_{C_t}^{-1}}\) and \(S_{C_t} = D_{N_{C_t}^{-1}}^{-1}\).

These quantities are computed in Listing [6](#org7278e18).

<figure id="org7278e18">
```{.python}
from theano.tensor.nlinalg import svd


y_tt = tt.specify_shape(tt.col(), [N_obs_tt, 1])
y_tt.name = 'y_t'


def filtering_step(y_t, m_tm1, U_C_tm1, S_C_tm1, F_t, G_t, N_W_t, N_V_t_inv):
    """Compute the sequential posterior state and prior predictive parameters."""

    # R_t = N_R.T.dot(N_R)
    N_R = tt.join(0,
                  matrix_dot(S_C_tm1, U_C_tm1.T, G_t.T),
                  N_W_t)
    # TODO: Consider an approach that only computes *one* set of singular
    # vectors
    _, s_R_t, U_R_t = svd(N_R)

    N_C_t = tt.join(0,
                    matrix_dot(N_V_t_inv, F_t.T, U_R_t),
                    tt.diag(tt_finite_inv(s_R_t)))
    _, d_N_C_t, Vt_N_C_t = svd(N_C_t)

    U_C_t, D_C_t = U_R_t.dot(Vt_N_C_t.T), tt.diag(tt_finite_inv(d_N_C_t))

    C_t = matrix_dot(U_C_t, D_C_t, U_C_t.T)

    a_t = G_t.dot(m_tm1)
    f_t = F_t.T.dot(a_t)
    m_t = a_t + matrix_dot(C_t, F_t, N_V_t_inv.T, N_V_t_inv, y_t - f_t)

    S_C_t = tt.sqrt(D_C_t)
    S_R_t = tt.diag(s_R_t)

    return [m_t, U_C_t, S_C_t, a_t, U_R_t, S_R_t]


U_C_0_tt, d_C_0_tt, _ = svd(C_0_tt)
S_C_0_tt = tt.diag(tt.sqrt(d_C_0_tt))

U_W_tt, d_W_tt, _ = svd(W_tt)
s_W_tt = tt.sqrt(d_W_tt)
N_W_tt = tt.diag(s_W_tt).dot(U_W_tt.T)

U_V_tt, D_V_tt, _ = svd(tt.as_tensor_variable(V_tt, ndim=2) if V_tt.ndim < 2 else V_tt)
S_V_inv_tt = tt.diag(tt_finite_inv(tt.sqrt(D_V_tt)))
N_V_inv_tt = S_V_inv_tt.dot(U_V_tt.T)


filter_res, filter_updates = theano.scan(fn=filtering_step,
                                         sequences=y_tt,
                                         outputs_info=[
                                             {"initial": m_0_tt, "taps": [-1]},
                                             {"initial": U_C_0_tt, "taps": [-1]},
                                             {"initial": S_C_0_tt, "taps": [-1]},
                                             {}, {}, {}  # a_t, U_R_t, S_R_t
                                         ],
                                         non_sequences=[F_tt, G_tt, N_W_tt, N_V_inv_tt],
                                         strict=True,
                                         name='theta_filtered')

(m_t, U_C_t, S_C_t, a_t, U_R_t, S_R_t) = filter_res
```
<figcaption>Listing 6</figcaption>
</figure>


## SVD-based Smoothing

We can use the ideas to produce SVD versions of the smoothing equations in \(\eqref{eq:dlm-smooth-moments}\). In this case, some extra steps are required in order to SVD-decompose \(S_t\) in the same manner as \(R_t\) and \(C_t^{-1}\) were.

First, notice that our target, \(S_t\), is a difference of matrices, unlike the matrix sums that comprised \(R_t\) and \(C_t^{-1}\) above. Furthermore, \(S_t\) is given as a difference of a (transformed) difference. To address the latter, we start by expanding \(S_t\) and setting \(B_t = C_t G_{t+1}^\top R_{t+1}^{-1}\) to obtain

\begin{equation}
  \begin{aligned}
    S_t &= C_t - B_t R_{t+1}^{-1} B_t^\top + B_t S_{t+1} B_t^\top
      \\
      &= H_t + B_t S_{t+1} B_t^\top
  \end{aligned}
  \label{eq:S_t_decomp}
\end{equation}

Having turned \(S_t\) into a sum of two terms, we can now consider another blocked SVD-based square-root reformulation, which starts with the reformulation of \(H_t\).

We can use the definition of \(R_t = G_{t_1} C_t G_{t+1}^\top + W_{t+1}\) to get

\begin{equation}
H_t = C_t - B_t \left(G_{t_1} C_t G_{t+1}^\top + W_{t+1}\right)^{-1} B_t^\top
.
\end{equation}

This form of \(H_t\) fits the Woodbury identity and results in \(H_t^{-1} = G_{t_1} W_{t+1}^{-1} G_{t+1}^\top + C_t^{-1}\), which is amenable to our square-root formulation.

Specifically, \(H_t^{-1} = N_{H_t}^{-\top} N_{H_t}^{-1}\), where

\begin{equation}
  \begin{aligned}
    N_{H_t}^{-1} &=
      \begin{pmatrix}
        N_{W_{t+1}}^{-1} G_{t+1}
        \\
        S_{C_t}^{-1} U_{C_t}
      \end{pmatrix}
  \end{aligned}
  .
  \label{eq:N_H_t_inv}
\end{equation}

From the SVD of \(N_{H_t}^{-1}\) we obtain the SVD of \(H_t\) as \(U_{H_t} = V_{N_{H_t}^{-1}}\) and \(D_{H_t} = {D_{N_{H_t}^{-1}}}^{-2} = S_{H_t}^2\).

Finally, using \(\eqref{eq:S_t_decomp}\) and \(\eqref{eq:N_H_t_inv}\) we can derive the last blocked square-root decomposition \(S_t = N_{S_t}^\top N_{S_t}\):

\begin{equation}
  \begin{aligned}
    N_{S_t} &=
      \begin{pmatrix}
        S_{H_t} U_{H_t}
        \\
        S_{C_{t+1}} U_{C_{t+1}}^\top B_t^\top
      \end{pmatrix}
  \end{aligned}
  .
  \label{eq:N_S_t}
\end{equation}

Again, we take the SVD of \(N_{S_t}\) and derive the SVD of \(S_t\) as \(U_{S_t} = V_{N_{S_t}}\) and \(D_{S_t} = D_{N_{S_t}}^2 = S_{S_t}^2\).

<figure id="org874cba4">
```{.python}

def smoother_step(m_t, U_C_t, S_C_t, a_tp1, U_R_tp1, S_R_tp1, s_tp1, U_S_tp1, S_S_tp1, G_tp1, N_W_tp1_inv):
    """Smooth a series starting from the "forward"/sequentially computed posterior moments."""

    N_C_t = S_C_t.dot(U_C_t.T)

    S_R_tp1_inv = tt_finite_inv(S_R_tp1)
    N_R_tp1_inv = S_R_tp1_inv.dot(U_R_tp1.T)

    # B_t = C_t @ G_tp1.T @ R_tp1
    B_t = matrix_dot(N_C_t.T, N_C_t, G_tp1.T, N_R_tp1_inv.T, N_R_tp1_inv)

    S_C_t_inv = tt_finite_inv(S_C_t)

    # N_H_t_inv.T @ N_H_t_inv = G_tp1 @ W_tp1_inv @ G_tp1.T + C_t_inv
    N_H_t_inv = tt.join(0,
                        N_W_tp1_inv.dot(G_tp1),
                        S_C_t_inv.dot(U_C_t.T))
    _, s_H_t_inv, U_H_t = svd(N_H_t_inv)

    # H_t = inv(N_H_t_inv.T @ N_H_t_inv) = C_t - B_t @ R_tp1 @ B_t.T
    D_H_t = tt.diag(tt_finite_inv(tt.square(s_H_t_inv)))

    # S_t = N_S_t.T.dot(N_S_t) = C_t - matrix_dot(B_t, R_tp1 - S_tp1, B_t.T)
    N_S_t = tt.join(0,
                     D_H_t.dot(U_H_t),
                     matrix_dot(S_S_tp1, U_S_tp1.T, B_t.T))
    U_S_t, d_S_t, _ = svd(N_S_t)

    S_S_t = tt.diag(d_S_t)

    s_t = m_t + B_t.dot(s_tp1 - a_tp1)

    return [s_t, U_S_t, S_S_t]


N_W_inv_tt = tt.diag(tt_finite_inv(s_W_tt)).dot(U_W_tt.T)

m_T = m_t[-1]
U_C_T = U_C_t[-1]
S_C_T = S_C_t[-1]

# These series only go from N_obs - 1 to 1
smoother_res, _ = theano.scan(fn=smoother_step,
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
                              name='theta_smoothed_obs')

(s_t_rev, U_S_t_rev, S_S_t_rev) = smoother_res

s_t = s_t_rev[::-1]
U_S_t = U_S_t_rev[::-1]
S_S_t = S_S_t_rev[::-1]

s_t = tt.join(0, s_t, [m_T])
U_S_t = tt.join(0, U_S_t, [U_C_T])
S_S_t = tt.join(0, S_S_t, [S_C_T])
```
<figcaption>Listing 7</figcaption>
</figure>


## Example

Listing [8](#orgad4c1a3) computes the filtered and smoothed means for our simulated series, and Figure [9](#org50d26f8) shows the results.

<figure id="orgad4c1a3">
```{.python}
filter_smooth_dlm = tt_function([y_tt, N_theta_tt, G_tt, F_tt],
                                [m_t, s_t],
                                # mode=theano.compile.mode.FAST_COMPILE
                                )

m_t_sim, s_t_sim = filter_smooth_dlm(y_sim, dlm_sim_values[N_theta_tt], dlm_sim_values[G_tt], dlm_sim_values[F_tt])
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
ax.plot(s_t_sim, label=r'$E[\theta_t \mid D_{T}]$', alpha=0.7, linewidth=0.8)
plt.legend(framealpha=0.4)
plt.tight_layout()
```
<figcaption>Listing 9</figcaption>
</figure>

<figure id="nil" class="plot"> ![Filtered and smoothed \(\theta_t\)&#x2013;against the true \(\theta_t\)&#x2013;computed using the SVD approach. \label{nil}]({attach}/articles/figures/svd-steps-sim-plot.png) <figcaption>Filtered and smoothed \(\theta_t\)&#x2013;against the true \(\theta_t\)&#x2013;computed using the SVD approach.</figcaption> </figure>


# Forward-filtering Backward-sampling

We can use the smoothing and filtering steps in the previous section to perform more efficient MCMC estimation than would otherwise be possible without the Rao-Blackwellization inherent to both steps.

Forward-filtering backward-sampling <a id="66fb99775f308e808a193bd7bb2d2038"><a href="#Fruhwirth-SchnatterDataaugmentationdynamic1994">(Fr\uhwirth-Schnatter 1994)</a></a> works by first computing the forward filtered moments, allowing one to draw \(\theta_T\) from \( \left(\theta_T \mid D_T\right) \sim \operatorname{N}\left(m_T, C_T\right) \) and, subsequently, \(\theta_t\) from \(\left(\theta_t \mid \theta_{t+1}, D_T \right) \sim \operatorname{N}\left(h_t, H_t\right)\).

The latter distribution's moments are easily derived from the filtered and smoothed moments:

\begin{equation}
  \begin{gathered}
    h_t = m_t + B_t \left(\theta_{t+1} - a_{t+1}\right)
    \\
    H_t = C_t - B_t R_{t+1} B^\top_t
  \end{gathered}
  \label{eq:ffbs-moments}
\end{equation}

Since all the quantities in \(\eqref{eq:ffbs-moments}\) appear in the filtering and smoothing moments, we can use the SVD-based approach described earlier to perform the updates and sampling. We reproduce the relevant subset of calculations in Listing [10](#org92c224d).

<figure id="org92c224d">
```{.python}
def ffbs_step(m_t, U_C_t, S_C_t, a_tp1, U_R_tp1, S_R_tp1, theta_tp1, F_tp1, G_tp1, N_W_tp1_inv, rng):
    """Perform forward-filtering backward-sampling."""

    S_C_t_inv = tt_finite_inv(S_C_t)

    # H_t_inv = N_H_t_inv.T @ N_H_t_inv = G_tp1 @ W_tp1_inv @ G_tp1.T + C_t_inv
    N_H_t_inv = tt.join(0,
                        N_W_tp1_inv.dot(G_tp1),
                        S_C_t_inv.dot(U_C_t.T))
    _, s_H_t_inv, U_H_t = svd(N_H_t_inv)

    D_H_t = tt.diag(tt_finite_inv(tt.square(s_H_t_inv)))

    N_C_t = S_C_t.dot(U_C_t.T)

    S_R_tp1_inv = tt_finite_inv(S_R_tp1)
    N_R_tp1_inv = S_R_tp1_inv.dot(U_R_tp1.T)

    # B_t = C_t @ G_tp1.T @ R_tp1
    B_t = matrix_dot(N_C_t.T, N_C_t, G_tp1.T, N_R_tp1_inv.T, N_R_tp1_inv)

    h_t = m_t + B_t.dot(theta_tp1 - a_tp1)
    h_t.name = 'h_t'

    # TODO: Add an option or optimization to use the SVD to sample in
    # `MvNormalRV`.
    # theta_t = MvNormalRV(h_t, H_t, rng=rng, name='theta_t_ffbs')
    theta_t = h_t + matrix_dot(U_H_t, tt.sqrt(D_H_t),
                               MvNormalRV(tt.zeros_like(h_t),
                                          tt.eye(h_t.shape[0]),
                                          rng=rng))

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
<figcaption>Listing 10</figcaption>
</figure>


## Example

Quantities besides the state values, \(\theta_t\), can be sampled sequentially (i.e. within the function `ffbs_step` in Listing [10](#org92c224d)), or after FFBS when all \(\theta_t \mid D_T\) have been sampled. These quantities can use the conditionally normal form of \(\left(\theta_t \mid \theta_{t+1}, D_T \right)\) to derive Gibbs steps, further Rao-Blackwellize hierarchical quantities, or apply any other means of producing posterior samples conditional on \(\left(\theta_t \mid \theta_{t+1}, D_T \right)\).

In this example, we will augment our original model by adding the classic gamma priors to our previously fixed state and observation scale parameters, \(\phi_W\) and \(\phi_V\), respectively.

This classical conjugate prior allows one to derive simple closed-form posteriors for a Gibbs sampler conditional on \(\theta_t \mid D_T\). Those posterior computations are defined in Listing [11](#org52662b2) and used to update the shared Theano variables for \(\phi_W\) and \(\phi_V\) within a Gibbs sampling loop in Listing [12](#org66be288).

<figure id="org52662b2">
```{.python}
# E[phi_W[0]] = 2.0, Var[phi_W[0]] = 50.0
# E[phi_W[1]] = 5.0, Var[phi_W[1]] = 50.0
phi_W_a, phi_W_b = np.r_[2.0**2 / 50.0, 5.0**2 / 50.0], np.r_[2.0 / 50.0, 5.00 / 50.0]

phi_V_a, phi_V_b = 0.5, 1.0

phi_W_post_tt = GammaRV(phi_W_a + N_obs_tt * 0.5,
                        phi_W_b + 0.5 * tt.square(theta_t_diff_rev).sum(0),
                        rng=rng_tt, name='phi_W_post')

phi_V_post_tt = GammaRV(phi_V_a + N_obs_tt * 0.5,
                        phi_V_b + 0.5 * tt.square(y_tt - f_t_post).sum(),
                        rng=rng_tt, name='phi_V_post')
```
<figcaption>Listing 11</figcaption>
</figure>

<figure id="org66be288">
```{.python}
ffbs_dlm = tt_function([y_tt, N_obs_tt, N_theta_tt, G_tt, F_tt],
                       [theta_t_post, phi_W_post_tt, phi_V_post_tt],
                       updates=ffbs_updates)

phi_W_tt.set_value(np.random.gamma(phi_W_a, scale=1.0/phi_W_b))
phi_V_tt.set_value(np.random.gamma(phi_V_a, scale=1.0/phi_V_b))

chain = 0
theta_label = r'$\theta_t \mid D_T$'
phi_W_label = r'$\phi_W \mid D_T$'
phi_V_label = r'$\phi_V \mid D_T$'
posterior_samples = {theta_label: [[]], phi_W_label: [[]], phi_V_label: [[]]}

for i in range(1000):

    theta_t_post_sim, phi_W_post_sim, phi_V_post_sim = ffbs_dlm(
        y_sim,
        dlm_sim_values[N_obs_tt], dlm_sim_values[N_theta_tt],
        dlm_sim_values[G_tt], dlm_sim_values[F_tt])

    # Update variance scale parameters
    phi_W_tt.set_value(phi_W_post_sim)
    phi_V_tt.set_value(phi_V_post_sim)

    posterior_samples[theta_label][chain].append(theta_t_post_sim)
    posterior_samples[phi_W_label][chain].append(phi_W_post_sim)
    posterior_samples[phi_V_label][chain].append(phi_V_post_sim)

    print(f'i={i},\tphi_W={phi_W_post_sim},\tphi_V={phi_V_post_sim}')

posterior_samples = {k: np.asarray(v) for k,v in posterior_samples.items()}
```
<figcaption>Listing 12</figcaption>
</figure>

Figure [13](#orge22cb90) shows the posterior \(\theta_t\) samples and Figure [14](#org9a3f957) plots the posterior sample traces.

<figure id="orge22cb90">
```{.python}
from cycler import cycler
from matplotlib.collections import LineCollection


plt.clf()

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.autoscale(enable=False)

# bivariate_cycler =  cycler('linestyle', ['-', '--']) * plt_orig_cycler
# ax.set_prop_cycle(bivariate_cycler)

thetas_shape = posterior_samples[theta_label][0].shape

cycle = ax._get_lines.prop_cycler

for d in range(thetas_shape[-1]):

    styles = next(cycle)
    thetas = posterior_samples[theta_label][0].T[d].T

    theta_lines = np.empty(thetas_shape[:-1] + (2,))
    theta_lines.T[0] = np.tile(np.arange(thetas_shape[-2]), [thetas_shape[-3], 1]).T
    theta_lines.T[1] = thetas.T

    ax.add_collection(
        LineCollection(theta_lines,
                       label=theta_label,
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
<figcaption>Listing 13</figcaption>
</figure>

<figure id="nil" class="plot"> ![Posterior \(\theta_t\) samples generated by a FFBS-based Gibbs sampler. \label{nil}]({attach}/articles/figures/ffbs-sim-plot.png) <figcaption>Posterior \(\theta_t\) samples generated by a FFBS-based Gibbs sampler.</figcaption> </figure>

<figure id="org9a3f957">
```{.python}
import arviz as az

az_trace = az.from_dict(posterior=posterior_samples)
az.plot_trace(az_trace, compact=True)
```
<figcaption>Listing 14</figcaption>
</figure>

<figure id="nil" class="plot"> ![Posterior sample traces for the FFBS-based Gibbs sampler. \label{nil}]({attach}/articles/figures/ffbs-trace-plot.png) <figcaption>Posterior sample traces for the FFBS-based Gibbs sampler.</figcaption> </figure>


# Discussion

So far, we've only shown how to perform FFBS for DLMs in Theano&#x2026;

# Bibliography
<a id="harrison_bayesian_1999"></a> Harrison & West, Bayesian Forecasting & Dynamic Models, Springer (1999). [↩](#4bbd465b4e78e5c5151b0cbba54d984e)

<a id="ZhangFixedintervalsmoothingalgorithm1996"></a> Zhang & Li, Fixed-Interval Smoothing Algorithm Based on Singular Value Decomposition, 916-921, in in: , Proceedings of the 1996 IEEE International Conference on Control Applications, 1996, edited by (1996) [↩](#0ae04c048b20d07f32d7f0f75bb51483)

<a id="PetrisDynamiclinearmodels2009"></a> Petris, Petrone & Campagnoli, Dynamic Linear Models, Springer (2009). [↩](#3a4d89388a434d7b1b91dc8690f3a03b)

<a id="Fruhwirth-SchnatterDataaugmentationdynamic1994"></a> Fr\"uhwirth-Schnatter, Data Augmentation and Dynamic Linear Models, <i>Journal of time series analysis</i>, <b>15(2)</b>, 183-202 (1994). <a href="http://onlinelibrary.wiley.com/doi/10.1111/j.1467-9892.1994.tb00184.x/abstract">link</a>. [↩](#66fb99775f308e808a193bd7bb2d2038)
