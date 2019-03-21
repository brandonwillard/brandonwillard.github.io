---
modified: '2019-1-7'
tags: 'hy,relational programming,python'
title: Readable Strings and Relational Programming in Hy
date: '2018-12-20'
author: 'Brandon T. Willard'
figure_dir: '{attach}/articles/figures/'
figure_ext: png
---

<div class="abstract">
Just some thoughts on a generalized `repr` for Hy and some connections with relational programming.

</div>


# Introduction

In the past few months, I've been working on [Hy](https://github.com/hylang/hy) a lot. It's been great for translating symbolic computation ideas originating in the Lisp community or simply performing the generic meta-programming inherent to the subject.

One feature I've been missing the most is "readable" print-outs from the REPL. In this case, "readable" means "a string that can be `eval`'ed to [re-]produce the object it's meant to represent". [Python calls the function(s) that produce these strings "`repr`"s](https://docs.python.org/3/library/functions.html#repr) and provides a generic `repr` function&#x2013;with limited Python "readability" guarantees&#x2013;and a `__repr__` property for object/class-level customization.

<div class="example" markdown="">
```{#org1fcc4d3 .python}
test_obj = {"a": 1, "b": [2, 3]}

# Produce a readable string using `repr`
obj_repr_str = repr(test_obj)
print(obj_repr_str)

# Re-create the object from its readable string form
obj_from_repr = eval(obj_repr_str)
print(obj_from_repr)

print(test_obj == obj_from_repr)
```

```{#org636a628 .python}
{'a': 1, 'b': [2, 3]}
{'a': 1, 'b': [2, 3]}
True


```

</div>

There's already a `hy.contrib.hy-repr` module that gets most of the way there, but it doesn't implement the Python standard library's `reprlib.Repr`. The class `reprlib.Repr` implements limits for the display lengths of the strings it produces, and its source code provides a few standard library implementations of primitive object `repr`s&#x2013;which require only trivial changes to produce the desired Hy syntax.

For these reasons&#x2013;and an overall interest in using and translating more of the Python standard library to Hy&#x2013;I decided to try a quick refactoring of `hy.contrib.hy-repr` that implements `reprlib.Repr`.


# The Hy `repr` Problem(s)

The translation of Hy AST to string form is fairly straight-forward. In most cases, one only needs to change the `repr`s for Python primitives and basic function calls (e.g. from `func(1)` to `(func 1)`); however, changing just a couple lines in `repr`/`__repr__` functions for all the Python builtins is very annoying.

Furthermore, what about those custom object `__repr__` methods? While one might be able to manually patch most&#x2013;if not all&#x2013;of the (Python-implemented) standard library objects, there are far too many 3rd-party library `__repr__`s with exactly the same trivial function-call form that can't reasonably be patched.


## Some approaches

The first few things that come to mind when considering a more general approach to Python-to-Hy `__repr__` translation involve some use of the existing `repr` code. That might come in the form of string manipulation of `repr` output, which `hy.contrib.hy-repr` already does in some cases, or quite possibly some use of a `repr` function's source or code object.

The latter seems like it has the potential to be more thorough and far-reaching, but also considerably more involved and computationally inefficient. Unfortunately, similar things can be said about the regex approach. Although it does seem a little easier to implement and&#x2013;for limited cases&#x2013;efficient enough for most purposes, it also comes across as much more brittle.

Fortunately, the latter is unnecessary, because, when the existing `repr` output is Python readable, it can be parsed by `ast.parse`. The function `ast.parse` effectively handles the regex work and yields the bulk of information needed for a Hy `repr` string: the function name and its (positional and keyword) arguments.

<div class="example" markdown="">
Let's say we implement our own object and `repr`.

```{#orgcb48770 .hy}
(defclass TestClass [object]
  (defn --init-- [self arg1 arg2 &optional kwarg1 kwarg2]
    (setv self.arg1 arg1
          self.arg2 arg2
          self.kwarg1 kwarg1
          self.kwarg2 kwarg2))
  (defn --repr-- [self]
    (.format "TestClass({}, {}, kwarg1={}, kwarg2={})"
             #* (lfor a [self.arg1 self.arg2
                         self.kwarg1 self.kwarg2]
                      (repr a)))))

(setv test-obj (TestClass 1 {"a" 1 "b" 2} :kwarg1 1 :kwarg2 "ok"))
(print (repr test-obj))
```

```{#org2d96184 .python}
TestClass(1, {'a': 1, 'b': 2}, kwarg1=1, kwarg2='ok')
```

Since the results are readable, we can do the following:

```{#org382ba33 .hy}
(import ast astor)
(setv repr-ast (ast.parse (repr test-obj) :mode "eval"))
(print (astor.dump repr-ast))
```

```{#orgf08b55b .python}
Expression(
    body=Call(func=Name(id='TestClass'),
              args=[Num(n=1),
                    Dict(keys=[Str(s='a'), Str(s='b')],
                         values=[Num(n=1), Num(n=2)])],
              keywords=[keyword(arg='kwarg1', value=Num(n=1)),
                        keyword(arg='kwarg2', value=Str(s='ok'))]))
```

</div>


## A Generalized Hy `repr` Prototype

With existing `repr` output converted to Python AST by Python itself (using `ast.parse`), we can produce readable Hy strings from the resulting AST objects.

In this scenario, we need only be concerned with the conversion of Python AST into readable Hy strings. This works like an inverse to the Hy compiler: in other words, a Hy decompiler. For `repr` purposes, only function call statements and their arguments need to be decompiled. Unfortunately, function arguments can consist of arbitrary Python/Hy objects, and that's how the decompilation responsibilities start to expand. If we limit our scope to a reasonable subset of Python builtins/primitives, the results can still be quite effective, and won't require a complete decompiler.

On the down-side, if a Hy `repr` implementation overrides the built-in `repr`, then arguments in existing `repr`/`__repr__`s might already be converted by the overridden `repr`; however, the results from `ast.parse` will undo/discard those results. Even so, custom class `__repr__`s aren't guaranteed to use the built-in `repr` on their arguments, so attempts to salvage already-converted `repr` output are undeniably fraught with complications.

<div class="example" markdown="">
Working from the `repr`-produced AST above, I mocked-up a quick prototype for a generic Python-to-Hy conversion function.

```{#orga431a0a .hy}
(import ast)
(import builtins)

(import [hy.contrib.hy-repr [hy-repr :as -hy-repr]])

(defn ast-funcall-to-hy [ast-obj repr1
                         &optional [level 1]]
  "Turn Python `ast.Call` expressions into Hy `repr` strings.

XXX: Only a very minimal subset of Python-to-Hy AST is implemented.

This can be used to turn a \"readable\" `repr` result, via an actual \"read\" by
`ast.parse`, to Python AST then Hy AST.
"
  (assert (and (instance? ast.Expression ast-obj)
               (instance? ast.Call ast-obj.body)))
  (setv func-name (. ast-obj body func id))
  (setv eval-fn (fn [o]
                  (if (instance? ast.Name o)
                      o.id
                      (repr1 (ast.literal-eval o) (dec level)))))
  (setv func-args (lfor a (. ast-obj body args) (eval-fn a)))
  (setv func-kwargs (lfor k (. ast-obj body keywords)
                          (.format ":{} {}" k.arg (eval-fn k.value))))
  (.format "({})" (.join " " (+ [func-name] func-args func-kwargs))))


(setv test-ast (ast.parse "range(x, y, blah=1, bloh=\"ok\")" :mode "eval"))
(print (ast-funcall-to-hy test-ast (fn [x &rest y] (-hy-repr x))))
```

```{#org9d771a1 .hy}
(range x y :blah 1 :bloh "ok")
```

`ast-funcall-to-hy` is an extremely narrow decompiler that only handles readable function calls (represented by `ast.Call` nodes), but, as part of a fallback sequence in a Hy `repr` implementation, it's still pretty useful.

A function like `ast-funcall-to-hy` can be used in `repr` logic as follows:

```{#org80332ca .hy}
(defn hy-repr [x &optional [level 1] [-repr (fn [x &rest y] (-hy-repr x))]]
  "Use `builtin.repr` results to generate readable Hy `repr` strings for cases
we haven't covered explicitly.
"
  (try
    (setv s (builtins.repr x))
    (when (not (.startswith s "<"))
      (do
        (setv repr-ast (ast.parse s :mode "eval"))
        (setv s (ast-funcall-to-hy repr-ast -repr))))
    s
    (except [Exception]
      (.format "<{} instance at {}>" x.__class__.__name__ (id x)))))
```

Now, for the example class, `TestClass`, we can demonstrate automatic conversion of its Python `__repr__` implementation.

```{#org21d5bb7 .hy}
(setv test-ast (TestClass 1 {"a" 2 "b" 3} :kwarg1 1 :kwarg2 "ok"))
(print (.format "before: {}\nafter: {}"
                (repr test-ast)
                (hy-repr test-ast)))
```

```{#orgb59b9c2 .text}
before: TestClass(1, {'a': 2, 'b': 3}, kwarg1=1, kwarg2='ok')
after: (TestClass 1 {"a" 2  "b" 3} :kwarg1 1 :kwarg2 "ok")
```

</div>


# Relational Programming

While considering all this, I kept thinking about how nice it would be to have a "bijective" compiler; in other words, the existing Hy compiler, which translates Hy-to-Python, **and** a Python-to-Hy (de)compiler. With a Python-to-Hy AST compiler, we could more broadly convert Python AST output&#x2013;like the kind in our example above&#x2013;to a `repr`/readable string in Hy.

The idea isn't too crazy, especially since one can easily work backward from a lot of the logic in the existing Hy compiler. There will be some edge cases that result in non-bijective translations (i.e. some round-trip Hy/Python translations might only be **equivalent** and not exactly **equal**), but this isn't necessarily a blocking issue. Decisions regarding "canonical" or reduced forms of Hy/Python AST might be necessary, especially if the resulting AST is intended to be more human readable than not.

Perhaps what's more discouraging is the effort it would take to ensure that the compilation processes going both ways are&#x2013;and stay&#x2013;coherent during the course of development. For instance, when changes are made to the standard compilation process (i.e. Hy-to-Python), it's likely that changes and tests would also be needed for the other direction.

This is where a paradigm like relational programming is particularly appealing: it provides a language for defining&#x2013;and means for computing&#x2013;the maps

\begin{equation*}
  \text{Hy Syntax}
  \longleftrightarrow \text{Python AST}
  \longleftrightarrow \text{Python Syntax}
  \;
\end{equation*}

in a cohesive way.

My relational programming DSL of choice, [miniKanren](http://minikanren.org), already has an implementation in Hy: [`loghyc` (and to be formally known as `adderall`)](https://github.com/algernon/adderall). We've been using it to perform static code analysis and refactoring in the project [`hydiomatic`](https://github.com/hylang/hydiomatic), so there's also a precedent for parsing Hy syntax in a relational context.

The missing/next step would be to output Python AST (instead of more Hy forms, like `hydiomatic` produces, for example). In the following sections, we will construct a small relational Hy/Python compiler as a proof-of-concept.


## A Prototype Relational Compiler

Creating a bi-directional Hy/Python AST compiler in miniKanren involves the construction of goals "relating" the two AST forms. For simplicity, we'll just consider function call expressions, like `func(args)` and `(func args)`.

<div class="remark" markdown="">
Also, since these kinds of relations are more easy to specify using constraints and subtle unification adjustments, we'll use a prototype microKanren implementation in Hy that provides immediate access to those: [`hypoKanren`](https://github.com/brandonwillard/hypoKanren).

Regardless, given the universality of miniKanren, the goals we construct should be directly translate-able to other implementations of miniKanren (even in completely different host languages).

The only obvious caveat to such translation is the availability of traditional `cons` semantics in the host language (i.e. the standard Lisp behavior of `cons`, `car`, `cdr`, and improper lists/`cons` pairs).

</div>

```{#orgc53b310 .hy}
(import ast)
(import astor)
(import types)
(import [collections [Callable]])

(import hy.models)
(import [hy.compiler [asty hy-eval hy-compile]])

(import [hypoKanren.goals [*]])
(import [hypoKanren.core [*]])


(require [hy.contrib.walk [let]])
(require [hypoKanren.goals [*]])
(require [hypoKanren.core [*]])
```

First, let's examine the general structure of the Python AST output generated by the Hy compiler for the Hy function-call given by `` `(func x :y z) ``.

```{#org7521929 .hy}
(astor.dump (hy-compile `(func x :y z) "__console__"))
```

```{#orgb137229 .python}
Module(
    body=[Expr(value=Call(func=Name(id='func'), args=[Name(id='x')], keywords=[keyword(arg='y', value=Name(id='z'))]))])
```

In what follows, we'll exclude the `ast.Module` and focus only on the `src.Expr` and its children.


### AST Object Unification

To make existing Python AST objects amenable to the [unification](https://en.wikipedia.org/wiki/Unification_(computer_science)) used by miniKanren, we implement `unify` specializations for `ast.AST` types. Our implementation simply generates unevaluated Hy forms, or Hy AST, that&#x2013;when evaluated&#x2013;would (re)create the `ast.AST` objects.

<div class="REMARK">
Alternatively, we could only ever use and create unevaluated Hy forms for Python AST. Providing unification for AST objects allows for more immediate integration with existing Python code and/or what it would most likely produce.

</div>

`hypoKanren` uses [`multipledispatch`](https://github.com/mrocklin/multipledispatch), so augmenting the unification process is easy. This is how we'll add support for AST objects.

<div class="REMARK">
There's already a good pure Python library for unification built upon `multipledispatch`, [`unfication`](https://github.com/mrocklin/unification). At a later time, it might be worthwhile to simply add support for Hy objects and use that library instead.

</div>

```{#org1468291 .hy}
(import [multipledispatch [dispatch]])
(import [hypoKanren.unify [*]])
(import [hy.models [*]])
(import [hy.contrib.walk [prewalk]])


(defmacro/g! dispatch-unify-trans [disp-type trans-func &optional [func 'unify]]
  `(do
     #@((dispatch ~disp-type object object)
        (defn unify-post-walk [~g!u ~g!v ~g!s]
          (~func (~trans-func ~g!u) ~g!v ~g!s)))
     #@((dispatch object ~disp-type object)
        (defn unify-post-walk [~g!u ~g!v ~g!s]
          (~func ~g!u (~trans-func ~g!v) ~g!s)))
     #@((dispatch ~disp-type ~disp-type object)
        (defn unify-post-walk [~g!u ~g!v ~g!s]
          (~func (~trans-func ~g!u) (~trans-func ~g!v) ~g!s)))))

(defn py-ast-to-expr [x]
  (defn -py-ast-to-expr [u]
    (setv ast-expr
          `(~(HySymbol (+ "ast." (name (type u))))
            ~@(chain.from-iterable
                (lfor f u.-fields
                      :if (hasattr u f)
                      [(HyKeyword f) (getattr u f)]))))
    ast-expr)
  (prewalk (fn [y] (if (instance? ast.AST y)
                       (-py-ast-to-expr y)
                       y))
           x))

;; Python AST expansion pre-unification
(dispatch-unify-trans ast.AST (fn [x] (py-ast-to-expr x)))
```

<div class="example" markdown="">
```{#org70c1279 .hy}
;; One is an `ast.AST` object, the other an unevaluated `ast.AST`
;; object-generating form.
(setv unify-exa-1 (unify (ast.Expr :value [])
                         `(ast.Expr :value ~(var 0))
                         {}))

;; Both are `ast.AST` objects
(setv unify-exa-2 (unify (ast.Expr :value [])
                         (ast.Expr :value (var 0))
                         {}))

(= (.get unify-exa-1 (var 0))
   (.get unify-exa-2 (var 0))
   [])
```

```{#orgc9b3e95 .python}
True
```

Listing [16](#org70c1279) illustrates unification of two `ast.AST` forms. The `(var 0)` objects are "logic variables" taking the value of sub-expressions that cause the two `unify` arguments to, well, unify. The third argument to `unify` is simply a `dict` that stores the logic variable/sub-expression mappings.

In other words, logic variables are like unknowns that `unify(u, v, s)` will "solve" in order to make `u` and `v` equal.

</div>

<div class="example" markdown="">
```{#orgd9478e0 .hy}
(unify (cons 'ast.Expr (var 0))
       (ast.Expr :value [(ast.Name :id "a")])
       {})
```

```{#org3302cb2 .python}
{(LVar 0): HyExpression([
  HyKeyword('value'),
  [HyExpression([
    HySymbol('ast.Name'),
    HyKeyword('id'),
    'a'])]])}
```

Listing [18](#orgd9478e0) is a more interesting example that demonstrates partial/improper list unification. Since `ast.AST` objects are expanded into equal object-instantiating Hy AST forms, `(cons 'ast.Expr (var 0))` is ultimately unified with a `HyExpression` (a subclass of `list`). Under the `cons` abstraction, `(var 0)` can be anything that&#x2013;when `cons`ed with the symbol `ast.Expr`&#x2013;will produce the expression `(ast.Expr :value [(ast.Name :id "a")])`. The result is the partial `HyExpression` comprising the arguments to the `ast.Expr` constructor&#x2013;in other words, the `cdr` of the `ast.AST` form.

</div>

We will also need to unify some limited Hy AST forms; specifically, `HySymbol`s. We will want to extract only the name part of a Hy symbol and relate that to Python `ast.Name`s via one of the latter's constructor arguments.

Similar to Python AST nodes, we will expand/lift/abstract `HySymbol`s to Hy expressions that&#x2013;when `eval`'ed&#x2013;would construct them. We can only do this in very limited cases; otherwise, we could end up producing ever-expanding forms.

```{#org7b1a1a0 .hy}
;; Hy AST expansion pre-unification
(defn unify-hysymbol [u v s]
  (cond
    [(= (first v) 'HySymbol)
     (print )
     (unify `(HySymbol ~(name u)) v s)]
    [True
     (unify u v s)]))

#@((dispatch HySymbol HyExpression object)
   (defn unify-post-walk [u v s]
     (unify-hysymbol u v s)))

#@((dispatch HyExpression HySymbol object)
   (defn unify-post-walk [u v s]
     (unify-hysymbol v u s)))
```

<div class="example" markdown="">
```{#orgbc4dbe9 .hy}
(unify 'a `(HySymbol ~(var 0)) {})
```

```{#orga00b5a1 .python}
{(LVar 0): 'a'}
```

Listing [20](#org7b1a1a0) demonstrates the expansion and unification of Hy AST symbols.

</div>


### Call-expression Goals

Next, we create the miniKanren goals that encapsulate the relationships between simple Hy and Python AST forms. In particular, we'll limit ourselves to only variable reference and function call forms.

```{#orge442cd6 .hy}
(defn listo [l]
  "A goal stating that `l` is a list."
  (conde
    [(== l []) s#]
    [(fresh [lcar lcdr]
            (== l (cons lcar lcdr))
            (listo lcdr))]
    [s# u#]))
```

The first AST relation is a simple one between `HySymbol`s and `ast.Name`s. This is where the `HySymbol` unification implemented above is used.

```{#org9fe6aea .hy}
(defn hy-py-symbolo [hy-ast py-ast]
  "A goal relating Hy and Python AST symbol/name objects (e.g. variable and
 function references)."
  (fresh [symbol-name py-ctx]
         (== hy-ast `(HySymbol ~symbol-name))
         (== py-ast `(ast.Name :id ~symbol-name
                               :ctx (ast.Load)))))
```

Some Python `ast.AST` types have fields consisting of lists containing other `ast.AST` objects (e.g. the `ast.Call` expressions below). We need a goal that enforces a relation between the Hy and Python AST forms of each element in such lists.

```{#org988ece9 .hy}
(defn lapplyo [func l-in l-out]
  "A goal that applies the goal `func` between all elements in lists `l-in` and
 `l-out`."
  (conj+
    (listo l-in)
    (conde
      [(fresh [lcar lcdr lout-car lout-cdr]
              (== l-in (cons lcar lcdr))
              (func lcar lout-car)
              (lapplyo func lcdr lout-cdr)
              (== l-out (cons lout-car lout-cdr)))]
      [(== l-in [])
       (== l-out l-in)])))
```

Finally, we create a goal for the AST of call expressions like `func(x, y, z)` and `(func x y z)`.

```{#org10bec84 .hy}
(defn hy-py-callo [hy-ast py-ast]
  "A goal relating call expressions in Python and Hy AST."
  (fresh [hy-op hy-args py-op py-args]
         ;; Hy AST form
         (== (cons hy-op hy-args) hy-ast)
         ;; Py AST form
         (== py-ast `(ast.Expr :value
                               (ast.Call :func
                                         ~py-op
                                         :args
                                         ~py-args
                                         :keywords
                                         [])))
         ;; These two must be related symbols
         (hy-py-symbolo hy-op py-op)
         ;; The arguments are related lists containing more of each AST type.
         (lapplyo hy-py-asto hy-args py-args)))

(defn hy-py-asto [hy-ast py-ast]
  "A goal for a 'branching' relation between multiple types of forms and their
 corresponding Python AST."
  (conde
    [(hy-py-symbolo hy-ast py-ast)]
    [(hy-py-callo hy-ast py-ast)]))
```

<div class="example" markdown="">
To demonstrate our [extremely] minimal relational compiler, we create a Hy function call expression and its corresponding Python AST.

```{#org7845670 .hy}
(setv hy-ast-exa `(print x y z))
(setv py-ast-exa (. (hy-compile hy-ast-exa "__console__") body [0]))
(.format "hy_ast_exa = {}\npy_ast_exa = {}"
         hy-ast-exa
         (astor.dump py-ast-exa))
```

```{#org09b193c .python}
hy_ast_exa = HyExpression([
  HySymbol('print'),
  HySymbol('x'),
  HySymbol('y'),
  HySymbol('z')])
py_ast_exa = Expr(value=Call(func=Name(id='print'), args=[Name(id='x'), Name(id='y'), Name(id='z')], keywords=[]))
```

We first run the Hy-to-Python direction by providing `hy-expro` the `hy-ast-exa` value above and a logic variable (i.e. an "unknown") for the Python AST term.

```{#org5336211 .hy}
(setv rel-res (run 1 [py-ast] (hy-py-asto hy-ast-exa py-ast)))
(setv ast-res (get rel-res 0 0))
ast-res
```

```{#org63b2bde .python}
HyExpression([
  HySymbol('ast.Expr'),
  HyKeyword('value'),
  HyExpression([
    HySymbol('ast.Call'),
    HyKeyword('func'),
    HyExpression([
      HySymbol('ast.Name'),
      HyKeyword('id'),
      'print',
      HyKeyword('ctx'),
      HyExpression([
        HySymbol('ast.Load')])]),
    HyKeyword('args'),
    HyExpression([
      HyExpression([
        HySymbol('ast.Name'),
        HyKeyword('id'),
        'x',
        HyKeyword('ctx'),
        HyExpression([
          HySymbol('ast.Load')])]),
      HyExpression([
        HySymbol('ast.Name'),
        HyKeyword('id'),
        'y',
        HyKeyword('ctx'),
        HyExpression([
          HySymbol('ast.Load')])]),
      HyExpression([
        HySymbol('ast.Name'),
        HyKeyword('id'),
        'z',
        HyKeyword('ctx'),
        HyExpression([
          HySymbol('ast.Load')])])]),
    HyKeyword('keywords'),
    HyList()])])
```

And, now, the other direction (i.e. known Python AST, unknown Hy AST).

```{#org7148809 .hy}
(setv rel-res (run 1 [hy-ast] (hy-py-asto hy-ast py-ast-exa)))
(setv ast-res (get rel-res 0 0))
ast-res
```

```{#orgbf40b7d .python}
[HyExpression([
  HySymbol('HySymbol'),
  'print']), HyExpression([
  HySymbol('HySymbol'),
  'x']), HyExpression([
  HySymbol('HySymbol'),
  'y']), HyExpression([
  HySymbol('HySymbol'),
  'z'])]
```

</div>
