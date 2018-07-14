---
author:
- Brandon Willard
date: '2016-10-27'
figure_dir: '{attach}/articles/figures/'
figure_ext: png
title: SymPy Expression Tree Manipulation
---

I’ve been working on some extensions to our special function computations in [Prediction risk for global-local shrinkage regression](https://arxiv.org/abs/1605.04796) and decided to employ [SymPy](https://github.com/sympy/sympy) as much as possible. Out of this came an [implementation](https://bitbucket.org/bayes-horseshoe-plus/hsplus-python-pkg/src/master/hsplus/horn_symbolic.py) of a bivariate confluent hypergeometric function: the [Humbert](https://en.wikipedia.org/wiki/Humbert_series) $\Phi_1$. This, and some numeric implementations, are available in a [Python package](https://bitbucket.org/bayes-horseshoe-plus/hsplus-python-pkg) and an [R package](https://bitbucket.org/bayes-horseshoe-plus/hsplus-r-pkg).

In the course of this work there are expectations that appear as ratios of $\Phi_1$ functions, so it’s helpful to have a symbolic replacement routine to identify them. [Pattern matching](http://docs.sympy.org/dev/modules/core.html#sympy.core.basic.Basic.match), [finding](http://docs.sympy.org/dev/modules/core.html#sympy.core.basic.Basic.find), substitution and [replacement](http://docs.sympy.org/dev/modules/core.html#sympy.core.basic.Basic.replace) are fairly standard in SymPy, so nothing special there; however, when you want something specific, it can get rather tricky.

Personally, I’ve found the approach offered by the [`sympy.strategies`](https://github.com/sympy/sympy/tree/master/sympy/strategies) and [`sympy.unify`](https://github.com/sympy/sympy/tree/master/sympy/unify) frameworks the most appealing. See the original discussion [here](https://groups.google.com/d/msg/sympy/fspCavhbd9I/vrzUitvgiuYJ). The reason for their appeal is mostly due to their organization of the processes behind expression tree traversal and manipulation. It’s much easier to see how a very specific and non-trivial simplification or replacement could be accomplished and iteratively improved. These points are made very well in the posts [here](http://matthewrocklin.com/blog/tags.html#SymPy-ref), so check them out.

Let’s say we want to write a function `as_expectations` that takes a `sympy.Expr` and replaces ratios of $\Phi_1$ functions according to the following pattern: $$\begin{equation}
E[X^n] = \frac{\Phi_1(\alpha, \beta, \gamma + n; x, y)}{\Phi_1(\alpha, \beta, \gamma; x, y)}
\;.
\label{eq:expectation}
\end{equation}$$

As an example, let’s set up a situation in which `as_expectations` would be used, and, from there, attempt to construct our function. Naturally, this will involve a test expression with terms that we know match Equation $\eqref{eq:expectation}$:

``` python
import sympy as sp

from hsplus.horn_symbolic import HornPhi1

a, b, g, z_1, z_2 = sp.symbols('a, b, g, z_1, z_2', real=True)
phi1_1 = HornPhi1((a, b), (g,), z_1, z_2)

n = sp.Dummy('n', integer=True, positive=True)
i = sp.Dummy('i', integer=True, nonnegative=True)

phi1_2 = HornPhi1((a, b), (g + n,), z_1, z_2)
phi1_3 = HornPhi1((a, b), (g + n - i,), z_1, z_2)

r_1 = phi1_2/phi1_1
r_2 = phi1_3/phi1_1

expr = a * r_1 - b * r_1 / g + sp.Sum(z_1/z_2 * r_2 - 3 * r_2, (i, 0,
n))
```

Our test expression `expr` looks like this

``` python
print(sp.latex(expr, mode='equation*', itex=True))
```

$$\begin{equation*}
\frac{a \operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad n + g,
\quad z_{1}, \quad z_{2}\right
)\right)}}{\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g,
\quad z_{1}, \quad z_{2}\right )\right)}} + \sum_{i=0}^{n}
\left(\frac{z_{1} \operatorname{\Phi_1}{\left(\left ( a, \quad b,
\quad - i + n + g, \quad z_{1}, \quad z_{2}\right )\right)}}{z_{2}
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g, \quad z_{1},
\quad z_{2}\right )\right)}} - \frac{3
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad - i + n + g,
\quad z_{1}, \quad z_{2}\right
)\right)}}{\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g,
\quad z_{1}, \quad z_{2}\right )\right)}}\right) - \frac{b
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad n + g, \quad
z_{1}, \quad z_{2}\right )\right)}}{g
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g, \quad z_{1},
\quad z_{2}\right )\right)}}
\end{equation*}$$

The ratios `r_1` and `r_2` should both be replaced by a symbol for $E[X^m]$, for $m = n$ and $m = n - i$ when $i < n$ respectively. We could allow $E[X^0]$, I suppose, but–for a more interesting discussion–let’s not.

We start by creating a SymPy pattern that expresses the mathematical form of $E[X^m]$ in Equation $\eqref{eq:expectation}$.

``` python
pnames = ('a', 'b', 'g', 'z_1', 'z_2')
phi1_wild_args_n = sp.symbols(','.join(n_ + '_w' for n_ in pnames),
                              cls=sp.Wild, real=True)

n_w = sp.Wild('n_w',
              properties=(lambda x: x.is_integer and x.is_positive,),
              exclude=(phi1_wild_args_n[2],))

phi1_wild_d = HornPhi1(phi1_wild_args_n[0:2],
                       phi1_wild_args_n[2:3],
                       *phi1_wild_args_n[3:5])

phi1_wild_n = HornPhi1(phi1_wild_args_n[0:2],
                       (phi1_wild_args_n[2] + n_w,),
                       *phi1_wild_args_n[3:5])

C_w = sp.Wild('C_w', exclude=[sp.S.Zero])
E_pattern = phi1_wild_n / phi1_wild_d

E_fn = sp.Function("E", real=True)
```

When we find an $E[X^m]$ we’ll replace it with the symbolic function `E_fn`.

If we focus on only one of the terms (one we know matches `E_pattern`), `r_1`, we should find that our pattern suffices:

``` python
>>> r_1.match(E_pattern)
{n_w_: _n, z_2_w_: z_2, z_1_w_: z_1, a_w_: a, g_w_: g, b_w_: b}
```

However, building up to the complexity of `expr`, we see that a simple product doesn’t:

``` python
>>> (a * r_1).match(E_pattern)
```

Basically, the product has introduced some problems that arise from associativity. Here are the details for the root expression tree:

``` python
>>> (a * r_1).func
<class 'sympy.core.mul.Mul'>
>>> (a * r_1).args
(a, 1/HornPhi1(a, b, g, z_1, z_2), HornPhi1(a, b, _n + g, z_1, z_2))
```

The root operation is multiplication and the operation’s arguments are all terms in the product/division.

Any complete search for matches to `E_pattern` would have to consider all possible combinations of terms in `(a * r_1).args`, i.e. all possible groupings that arise due to associativity. The simple inclusion of another `Wild` term causes the match to succeed, since SymPy’s basic pattern matching does account for associativity in this case.

Here are a few explicit ways to make the match work:

``` python
>>> (a * r_1).match(C_w * E_pattern)
{a_w_: a, n_w_: _n, g_w_: g, z_2_w_: z_2, C_w_: a, b_w_: b, z_1_w_:
z_1}
```

or as a replacement:

``` python
res = (a * r_1).replace(C_w * E_pattern, C_w * E_fn(n_w,
*phi1_wild_args_n))
print(sp.latex(res, mode='equation*', itex=True))
```

$$\begin{equation*}
a E{\left (n,a,b,g,z_{1},z_{2} \right )}
\end{equation*}$$

and via `rewriterule`:

``` python
from sympy.unify.rewrite import rewriterule
rl = rewriterule(C_w * E_pattern,
                 C_w * E_fn(n_w, *phi1_wild_args_n),
                 phi1_wild_args_n + (n_w, C_w))
res = list(rl(a * r_1))
print(sp.latex(res, mode='equation*', itex=True))
```

$$\begin{equation*}
\left [ a E{\left (n,a,b,g,z_{1},z_{2} \right )}\right ]
\end{equation*}$$

The advantage in using `rewriterule` is that multiple matches will be returned. If we add another $\Phi_1$ in the numerator, so there are multiple possible $E[X^m]$, we get

``` python
phi1_4 = HornPhi1((a, b), (g + n + 1,), z_1, z_2)

res = list(rl(a * r_1 * phi1_4))
print(sp.latex(res, mode='equation*', itex=True))
```

$$\begin{equation*}
\left [ a \operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad n +
g, \quad z_{1}, \quad z_{2}\right )\right)} E{\left (n +
1,a,b,g,z_{1},z_{2} \right )}, \quad a
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad n + g, \quad
z_{1}, \quad z_{2}\right )\right)} E{\left (n + 1,a,b,g,z_{1},z_{2}
\right )}, \quad a \operatorname{\Phi_1}{\left(\left ( a, \quad b,
\quad n + g + 1, \quad z_{1}, \quad z_{2}\right )\right)} E{\left
(n,a,b,g,z_{1},z_{2} \right )}, \quad a
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad n + g, \quad
z_{1}, \quad z_{2}\right )\right)} E{\left (n + 1,a,b,g,z_{1},z_{2}
\right )}, \quad a \operatorname{\Phi_1}{\left(\left ( a, \quad b,
\quad n + g, \quad z_{1}, \quad z_{2}\right )\right)} E{\left (n +
1,a,b,g,z_{1},z_{2} \right )}, \quad a
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad n + g + 1, \quad
z_{1}, \quad z_{2}\right )\right)} E{\left (n,a,b,g,z_{1},z_{2} \right
)}\right ]
\end{equation*}$$

FYI: the associativity of terms inside the function arguments is causing the seemingly duplicate results.

Naive use of `Expr.replace` doesn’t give all results; instead, it does something likely unexpected:

``` python
res = (a * r_1 * phi1_4).replace(C_w * E_pattern,
                                 C_w * E_fn(n_w, *phi1_wild_args_n))
print(sp.latex(res, mode='equation*', itex=True))
```

$$\begin{equation*}
a E{\left (n,a,b,g,z_{1},z_{2} \right )} E{\left (n +
1,a,b,g,z_{1},z_{2} \right )} \operatorname{\Phi_1}{\left(\left ( a,
\quad b, \quad g, \quad z_{1}, \quad z_{2}\right )\right)}
\end{equation*}$$

Returning to our more complicated `expr`…Just because we can match products doesn’t mean we’re finished, since we still need a good way to traverse the entire expression tree and match the sub-trees. More importantly, adding the multiplicative `Wild` term `C_w` is more of a hack than a direct solution, since we don’t want the matched contents of `C_w`.

Although `Expr.replace/xreplace` will match sub-expressions, we found above that it produces some odd results. Those results persist when applied to more complicated expressions:

``` python
res = expr.replace(C_w * E_pattern, C_w * E_fn(n_w,
*phi1_wild_args_n))
print(sp.latex(res, mode='equation*', itex=True))
```

$$\begin{equation*}
a E{\left (n,a,b,g,z_{1},z_{2} \right )} - \frac{b}{g} E{\left
(n,a,b,g,z_{1},z_{2} \right )} + \sum_{i=0}^{n} \left(\frac{z_{1}
E{\left (n,a,b,- i + g,z_{1},z_{2} \right )}
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad - i + g, \quad
z_{1}, \quad z_{2}\right )\right)}}{z_{2}
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g, \quad z_{1},
\quad z_{2}\right )\right)}} - \frac{3 E{\left (n,a,b,- i +
g,z_{1},z_{2} \right )} \operatorname{\Phi_1}{\left(\left ( a, \quad
b, \quad - i + g, \quad z_{1}, \quad z_{2}\right
)\right)}}{\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g,
\quad z_{1}, \quad z_{2}\right )\right)}}\right)
\end{equation*}$$

Again, it looks like the matching was a little too liberal and introduced extra `E` and `HornPhi1` terms. This is to be expected from the `Wild` matching in SymPy; it needs us to specify what *not* to match, as well. Our “fix” that introduced `C_w` is the exact source of the problem, but we can tell it not to match `HornPhi1` terms and get better results:

``` python
C_w = sp.Wild('C_w', exclude=[sp.S.Zero, HornPhi1])
res = expr.replace(C_w * E_pattern, C_w * E_fn(n_w,
*phi1_wild_args_n))
print(sp.latex(res, mode='equation*', itex=True))
```

$$\begin{equation*}
a E{\left (n,a,b,g,z_{1},z_{2} \right )} - \frac{b}{g} E{\left
(n,a,b,g,z_{1},z_{2} \right )} + \sum_{i=0}^{n} \left(\frac{z_{1}
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad - i + n + g,
\quad z_{1}, \quad z_{2}\right )\right)}}{z_{2}
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g, \quad z_{1},
\quad z_{2}\right )\right)}} - \frac{3
\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad - i + n + g,
\quad z_{1}, \quad z_{2}\right
)\right)}}{\operatorname{\Phi_1}{\left(\left ( a, \quad b, \quad g,
\quad z_{1}, \quad z_{2}\right )\right)}}\right)
\end{equation*}$$

We’ve stopped it from introducing those superfluous `E` terms, but we’re still not getting replacements for the `HornPhi1` ratios in the sums. Let’s single out those terms and see what’s going on:

``` python
res = r_2.find(C_w * E_pattern)
print(sp.latex(res, mode='equation*', itex=True))
```

$$\begin{equation*}
\left\{\right\}
\end{equation*}$$

The constrained integer `Wild` term, `n_w`, probably isn’t matching. Given the form of our pattern, `n_w` should match `n - i`, but `n - i` isn’t strictly positive, as required:

``` python
>>> (n - i).is_positive == True
False
>>> sp.ask(sp.Q.positive(n - i)) == True
False
```

Since $n > 0$ and $i >= 0$, the only missing piece is that $n > i$. The most relevant mechanism in SymPy to assess this information is the [`sympy.assumptions`](http://docs.sympy.org/dev/modules/assumptions/index.html) interface. We could add and retrieve the assumption `sympy.Q.is_true(n > i)` via `sympy.assume.global_assumptions`, or perform these operations inside of a Python `with` block, etc. This context management, via `sympy.assumptions.assume.AssumptionsContext`, would have to be performed manually, since I am not aware of any such mechanism offered by `Sum` and/or `Basic.replace`.

Unfortunately, these ideas sound good, but aren’t implemented:

``` python
>>> sp.ask(sp.Q.positive(n - i), sp.Q.is_true(n > i)) == True
False
```

See the documentation for `sympy.assumptions.ask.ask`; it explicitely states that inequalities aren’t handled, yet.

We could probably perform a manual reworking of `sympy.Q.is_true(n > i)` to `sympy.Q.is_true(n - i > 0)`, which is of course equivalent to `sympy.Q.positive(n - i)`: the result we want.

If one were to provide this functionality, there’s still the question of how the relevant `AssumptionsContext`s would be created and passed around/nested during the subexpression replacements. There is no apparent means of adding this sort of functionality through the `Basic.replace` interface, so this path looks less appealing. However, nesting `with` blocks from strategies in `sympy.strategies` does seem quite possible. For example, in `sympy.strategies.traverse.sall`, one could possibly wrap the `return` statement after the `map(rule, ...)` call in a `with sympy.assuming(...):` block that contains the assumptions for any variables arising as, say, the index of a `Sum`–like in our case. In this scenario, code in the subexpressions would be able to ask questions like `sympy.Q.is_true(n > i)` without altering the global assumptions context or the objects involved.

Anyway, that’s all I wanted to cover here. Perhaps later I’ll post a hack for the assumptions approach, but–at the very least–I’ll try to follow up with a more direct solution that uses `sympy.strategies`.
