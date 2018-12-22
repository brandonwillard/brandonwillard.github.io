---
author: 'Brandon T. Willard'

date: '2018-12-20'
figure_dir: '{attach}/articles/figures/'
figure_ext: png
title: Readable Strings and Relational Programming in Hy
---

<div class="abstract">
Just some thoughts on a generalized `repr` for Hy and some connections with relational programming.

</div>


# Introduction

In the past few months, I've been working on [Hy](https://github.com/hylang/hy) a lot. It's been great for translating symbolic computation ideas originating in the Lisp community or simply performing the generic meta-programming inherent to the subject.

One feature I've been missing the most is "readable" print-outs from the REPL. In this case, "readable" means "a string that can be `eval`'ed to [re-]produce the object it's meant to represent". [Python calls the function(s) that produce these strings "`repr`"s](https://docs.python.org/3/library/functions.html#repr) and provides a generic `repr` function&#x2013;with limited Python "readability" guarantees&#x2013;and a `__repr__` property for object/class-level customization.

<div class="example" markdown="" env-number="1">

```python
test_obj = {"a": 1, "b": [2, 3]}

# Produce a readable string using `repr`
obj_repr_str = repr(test_obj)
print(obj_repr_str)

# Re-create the object from its readable string form
obj_from_repr = eval(obj_repr_str)
print(obj_from_repr)

print(test_obj == obj_from_repr)
```

```python
{'a': 1, 'b': [2, 3]}
{'a': 1, 'b': [2, 3]}
True


```

</div>

There's already a `hy.contrib.hy-repr` module that gets most of the way there, but it doesn't implement the Python standard library's `reprlib.Repr`. The class `reprlib.Repr` implements limits for the display lengths of the strings it produces, and it's source code provides a few standard library implementations of primitive object `repr`s&#x2013;which require only trivial changes to produce the desired Hy syntax.

For these reasons&#x2013;and an overall interest in using and translating more of the Python standard library to Hy&#x2013;I decided to try a quick refactoring of `hy.contrib.hy-repr` that implements `reprlib.Repr`.


## The Problem(s)

The translation of Hy AST to string form is fairly straight-forward. In most cases, one only needs to change the `repr`s for Python primitives and basic function calls (e.g. from `func(1)` to `(func 1)`); however, changing just a couple lines in `repr`/`__repr__` functions for all the Python builtins is very annoying.

Furthermore, what about those custom object `__repr__` methods? While one might be able to manually patch most&#x2013;if not all&#x2013;of the (Python-implemented) standard library objects, there are far too many 3rd-party library `__repr__`s with exactly the same trivial function-call form that can't reasonably be patched.


## Some approaches

The first few things that come to mind when considering a more general approach to Python-to-Hy `__repr__` translation involve some use of the existing `repr` code. That might come in the form of string manipulation of `repr` output, which `hy.contrib.hy-repr` already does in some cases, or quite possibly some use of a `repr` function's source or code object.

The latter seems like it has the potential to be more thorough and far-reaching, but also considerably more involved and computationally inefficient. Unfortunately, similar things can be said about the regex approach. Although it does seem a little easier to implement and&#x2013;for limited cases&#x2013;efficient enough for most purposes, it also comes across as much more brittle.

Fortunately, the latter is unnecessary, because, when the existing `repr` output is Python readable, it can be parsed by `ast.parse`. The function `ast.parse` effectively handles the regex work and yields the bulk of information needed for a Hy `repr` string: the function name and its (positional and keyword) arguments.

<div class="example" markdown="" env-number="2">

Let's say we implement our own object and `repr`.

```hy
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

```python
TestClass(1, {'a': 1, 'b': 2}, kwarg1=1, kwarg2='ok')
```

Since the results are readable, we can do the following:

```hy
(import ast astor)
(setv repr-ast (ast.parse (repr test-obj) :mode "eval"))
(print (astor.dump repr-ast))
```

```python
Expression(
    body=Call(func=Name(id='TestClass'),
              args=[Num(n=1),
                    Dict(keys=[Str(s='a'), Str(s='b')],
                         values=[Num(n=1), Num(n=2)])],
              keywords=[keyword(arg='kwarg1', value=Num(n=1)),
                        keyword(arg='kwarg2', value=Str(s='ok'))]))
```

</div>


# An Implemented Partial Solution

With existing `repr` output converted to Python AST by Python itself (using `ast.parse`), we can produce readable Hy strings from the resulting AST objects.

In this scenario, we need only be concerned with the conversion of Python AST into readable Hy strings. This works like an inverse to the Hy compiler: in other words, a Hy decompiler. For `repr` purposes, only function call statements and their arguments need to be decompiled. Unfortunately, function arguments can consist of arbitrary Python/Hy objects, and that's how the decompilation responsibilities start to expand. If we limit our scope to a reasonable subset of Python builtins/primitives, the results can still be quite effective, and won't require a complete decompiler.

On the down-side, if a Hy `repr` implementation overrides the built-in `repr`, then arguments in existing `repr`/`__repr__`s might already be converted by the overridden `repr`; however, the results from `ast.parse` will undo/discard those results. Even so, custom class `__repr__`s aren't guaranteed to use the built-in `repr` on their arguments, so attempts to salvage already-converted `repr` output are undeniably fraught with complications.

<div class="example" markdown="" env-number="3">

Working from the `repr`-produced AST above, I mocked-up a quick prototype for a generic Python-to-Hy conversion function.

```hy
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

```hy
(range x y :blah 1 :bloh "ok")
```

`ast-funcall-to-hy` is an extremely narrow decompiler that only handles readable function calls (represented by `ast.Call` nodes), but, as part of a fallback sequence in a Hy `repr` implementation, it's still pretty useful.

A function like `ast-funcall-to-hy` can be used in `repr` logic as follows:

```hy
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

```hy
(setv test-ast (TestClass 1 {"a" 2 "b" 3} :kwarg1 1 :kwarg2 "ok"))
(print (.format "before: {}\nafter: {}"
                (repr test-ast)
                (hy-repr test-ast)))
```

```text
before: TestClass(1, {'a': 2, 'b': 3}, kwarg1=1, kwarg2='ok')
after: (TestClass 1 {"a" 2  "b" 3} :kwarg1 1 :kwarg2 "ok")
```

</div>


# A use for relational programming

While considering all this, I kept thinking about how nice it would be to have a "bijective" compiler; in other words, the existing Hy compiler, which translates Hy-to-Python, **and** a Python-to-Hy (de)compiler. With a Python-to-Hy AST compiler, we could more broadly convert Python AST output&#x2013;like the kind in our example above&#x2013;to a `repr`/readable string in Hy.

The idea isn't too crazy, especially since one can easily work backward from a lot of the logic in the existing Hy compiler. There will be some edge cases that result in non-bijective translations (i.e. some round-trip Hy/Python translations might only be **equal** and not exactly **equivalent**), but this isn't necessarily a blocking issue. Decisions regarding "canonical" or reduced forms of Hy/Python AST might be necessary, especially if the resulting AST is intended to be more human readable than not.

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

The missing/next step would be to output Python AST (instead of more Hy forms, like `hydiomatic` produces, for example).

Perhaps, in a follow-up, I'll illustrate how this can be done.
