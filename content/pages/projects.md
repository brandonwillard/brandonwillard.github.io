---
title: Projects
---

Check [Github](https://github.com/brandonwillard) and 
[Bitbucket](https://bitbucket.org/brandonwillard) to see what I've been up to.  Otherwise, here's
a quick roundup...

<div class='project' name="amimodels" markdown>
[code](https://github.com/openeemeter/amimodels)

[PyMC](https://pymc-devs.github.io/pymc/) models for time series modeling of
energy usage.  This project provides generic Hidden Markov Models and efficient
Rao-Blackwellized samplers.

</div>

<div class='project' name="MTA Bus Time" markdown>
[site](http://bustime.mta.info/),
[news](http://gothamist.com/2014/02/24/mtas_real_time_bus_tracking_info_ex.php),
[technology](http://bustime.mta.info/wiki/Main/Technology),
[code](https://github.com/camsys/onebusaway-nyc/commits?author=brandonwillard)

Bus Time is the open source Java suite that provides real-time MTA bus tracking
to NYC.  I designed and developed the statistical models and helped build the
production service components.  The model handles free and constrained
location tracking along street networks, inference for unobserved operational
states (e.g. in layover, at a stop, in progress) and path-based states (e.g.
current trip, route, run), as well as inference for faulty operator input (e.g.
operator ids, sign codes).

The model handles real-time updates at ~30 second intervals for hundreds of
routes and thousands of buses simultaneously.  Its statistical specification is
Bayesian and its estimation is performed by a custom particle filter.  Model
parameters are learned real-time and error estimates are available on the fly.
</div>

<div class='project' name="open-tracking-tools" markdown>
[code](https://github.com/brandonwillard/open-tracking-tools)

`open-tracking-tools` is an open-source vehicle tracking library that implements
standard and advanced Particle Filter approaches to model on/off-road states.
Paths traveled on road segments are inferred and model parameters (e.g.
on/off-road probabilities, GPS error) can be estimated real-time.
</div>

<div class='project' name="StatsLibExtensions" markdown>
[code](https://bitbucket.org/brandonwillard/statslibextensions)

Extensions to the [Cognitive Foundry API](https://github.com/algorithmfoundry/Foundry) 
including, but not limited to, specialized distributions, sampling techniques,
and numerically stable computations for Dynamic Linear Models.
</div>

<div class='project' name="ParticleBayes" markdown>
[code](https://bitbucket.org/brandonwillard/particlebayes)

`ParticleBayes` is an R package implementing a collection of particle filters
for hierarchical Bayesian models that perform sequential parameter estimation.
</div>

<div class='project' name="ParticleLearningModels" markdown>
[code](https://bitbucket.org/brandonwillard/particlelearningmodels)

Java code for Bayesian models that are estimated by Particle Filters and
implement parameter learning.
</div>

<div class='project' name="CTA-sim" markdown>
[site](http://dssg.io/projects/2013/\#cta),
[code](https://github.com/dssg/cta-sim)

Big data simulation of Chicago's public transportation to improve
transit planning and reduce bus crowding.
</div>

<div class='project' name="energywise" markdown>
[site](http://dssg.io/projects/2013/\#lbnl),
[code](https://github.com/dssg/energywise)

An energy analytics tool to make commercial building more energy efficient.
</div>

