---
title: Projects
---

Check [Github](https://github.com/brandonwillard) and
[Bitbucket](https://bitbucket.io/brandonwillard) to see what I've been up to.  Otherwise, here's
a quick roundup...

<div class='project' name="hsplus" markdown>
<ul class='project-links'>
  <li><a href="https://bitbucket.org/bayes-horseshoe-plus/hsplus-python-pkg">python-code</a></li>
  <li><a href="https://bitbucket.org/bayes-horseshoe-plus/hsplus-r-pkg">R-code</a></li>
</ul>

`hsplus` is a Python library (with R bindings) that provides estimates for
quantities involving the Horseshoe and Horseshoe+ shrinkage priors.
It also contains general numerical estimation procedures for bivariate confluent
hypergeometric functions, as well as symbolic SymPy implementations.

</div>

<div class='project' name="amimodels" markdown>
<ul class='project-links'>
  <li><a href="https://github.com/brandonwillard/amimodels">code</a></li>
  <li><a href="https://amimodels.readthedocs.io/en/latest/methodology.html">docs</a></li>
</ul>

`amimodels` is a Python library that provides core implementations of models
designed for use with Advanced Metering Infrastructure (AMI) data in
[`eemeter`](http://www.openeemeter.org/). The implementations
are fundamentally Bayesian state-space and mixture models that automatically
account for the systematic changes, missing data and varied observation
frequencies. The models and custom MCMC estimation methods are written in
[PyMC2](https://pymc-devs.github.io/pymc/) and--as such--are easily extensible.

</div>

<div class='project' name="estimation examples" markdown>
<ul class='project-links'>
  <li><a href="http://andrewgelman.com/2015/02/17/bayesian-survival-analysis-horseshoe-priors/">news</a></li>
</ul>

<br>
Complete R and [Stan](http://mc-stan.org/) Horseshoe and Horseshoe+ shrinkage prior examples from our paper
["Default Bayesian analysis with global-local shrinkage priors"](https://academic.oup.com/biomet/article-abstract/103/4/955/2659031/Default-Bayesian-analysis-with-global-local):
<ul>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-horseshoe-plus/horseshoe-stan-nvm.html">sparse multivariate normal estimation</a>
  </li>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-horseshoe-plus/horseshoe-stan-fc.html">ratio of normals estimation</a>
  </li>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-horseshoe-plus/horseshoe-stan-pm.html">product of normals estimation</a>
  </li>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-horseshoe-plus/horseshoe-stan-efron-max.html">max of multivariate normal estimation</a>
  </li>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-horseshoe-plus/horseshoe-stan-efron-sum.html">sum of squares estimation</a>
  </li>
</ul>

Sparse signal estimation examples, in R, using proximal algorithms:
<ul>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-map/bayes-map-l2-l1-example.html">$\ell_2$ loss with $\ell_1$ penalty</a>
  </li>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-map/bayes-map-l2-lq-example.html">$\ell_2$ loss with $\ell_q$, for $0 < q < 1$ penalty</a> (with C++ implementation)
  </li>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-map/bayes-map-l2-pareto-example.html">$\ell_2$ loss with double-Pareto penalty</a>
  </li>
  <li><a href="https://brandonwillard.bitbucket.io/bayes-map/bayes-map-logit-pareto-example.html">logit loss with double-Pareto penalty</a>
  </li>
</ul>

  [Simple, custom slice sampler](https://brandonwillard.bitbucket.io/bayes-optimization/bayes-opt-rosenbrock.html) for the
  [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).
</div>

<div class='project' name="MTA Bus Time" markdown>
<ul class='project-links'>
  <li><a href="http://bustime.mta.info/">site</a></li>
  <li><a href="http://gothamist.com/2014/02/24/mtas_real_time_bus_tracking_info_ex.php">news</a></li>
  <li><a href="http://bustime.mta.info/wiki/Main/Technology">tech</a></li>
  <li><a href="https://github.com/camsys/onebusaway-nyc/commits?author=brandonwillard">code</a></li>
</ul>

Bus Time is the open source Java suite that provides real-time bus tracking to
NYC.  I designed and developed the
[statistical inference capabilities](https://github.com/camsys/onebusaway-nyc/wiki/Inference-Engine) and
helped build the production service components.  The model handles free and
constrained location tracking along street networks, inference for unobserved
operational states (e.g. in layover, at a stop, in progress) and path-based
states (e.g.  current trip, route, run), as well as inference for faulty
operator input (e.g.  operator ids, sign codes).

In production the model handles real-time updates at ~30 second intervals for
hundreds of routes and thousands of buses simultaneously.  Its statistical
specification is Bayesian and its estimation is performed by a custom particle
filter.
</div>

<div class='project' name="prox-methods" markdown>
<ul class='project-links'>
  <li><a href="https://bitbucket.org/prox-methods-in-stats/prox-methods-rpkg">code</a></li>
</ul>

`prox-methods` is a very experimental R package with C++ implementations (via Rcpp) for some of the
proximal optimization methods from the paper
["Proximal Algorithms in Statistics and Machine Learning"](https://projecteuclid.org/euclid.ss/1449670858).

</div>

<div class='project' name="open-tracking-tools" markdown>
<ul class='project-links'>
  <li><a href="https://github.com/brandonwillard/open-tracking-tools">code</a></li>
</ul>

`open-tracking-tools` is an open-source vehicle tracking library that
implements custom Particle Filters to infer locations, paths and on/off-road
states.  Given a transit graph, `open-tracking-tools` provides robust real-time
Bayesian inference for noisy GPS data.
[OpenTripPlanner](http://www.opentripplanner.org/uses) graph support is built in,
so street information encoded in [OpenStreetMap](https://www.openstreetmap.org/) can
be used with fairly minimal effort.
</div>

<div class='project' name="StatsLibExtensions" markdown>
<ul class='project-links'>
  <li><a href="https://bitbucket.org/brandonwillard/statslibextensions">code</a></li>
</ul>

Extensions to the [Cognitive Foundry API](https://github.com/algorithmfoundry/Foundry)
including, but not limited to, specialized distributions, sampling techniques,
and numerically stable computations for Dynamic Linear Models.
</div>

<div class='project' name="ParticleBayes" markdown>
<ul class='project-links'>
  <li><a href="https://bitbucket.org/brandonwillard/particlebayes">code</a></li>
</ul>

`ParticleBayes` is an R package implementing a collection of particle filters
for hierarchical Bayesian models that perform sequential parameter estimation.
</div>

<div class='project' name="ParticleLearningModels" markdown>
<ul class='project-links'>
  <li><a href="https://bitbucket.org/brandonwillard/particlelearningmodels">code</a></li>
</ul>

Java code for Bayesian models that are estimated by Particle Filters and
implement parameter learning.
</div>

<div class='project' name="CTA-sim" markdown>
<ul class='project-links'>
  <li><a href="https://dssg.uchicago.edu/cta-why-bus-crowding-happens-and-how-data-can-help">description</a></li>
  <li><a href="https://dssg.uchicago.edu/project/simulating-better-bus-service/">site</a></li>
  <li><a href="https://github.com/dssg/cta-sim">code</a></li>
</ul>

Big data simulation of Chicago's public transportation to improve
transit planning and reduce bus crowding.
</div>

<div class='project' name="energywise" markdown>
<ul class='project-links'>
  <li><a href="https://github.com/dssg/energywise/wiki/Problem">description</a></li>
  <li><a href="https://dssg.uchicago.edu/project/predicting-building-energy-savings/">site</a></li>
  <li><a href="https://github.com/dssg/energywise">code</a></li>
</ul>

An energy analytics tool to make commercial building more energy efficient.
</div>
