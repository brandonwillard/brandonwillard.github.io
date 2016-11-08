r"""
Pandoc filter that doesn't throw away `\eqref{...}` in latex.
Other output formats are unaffected.

Example
=======

Let's create a test latex file:

    >>> test_file = r'''
    ... \\\maketitle
    ... \\\begin{equation}
    ... \\\label{eq1}
    ... hey
    ... \\\end{equation}
    ... yo yo \\\eqref{eq1}
    ... '''
    >>> !echo "$test_file" > test_eqref.tex

If you want to see what JSON/parsed input our filter is getting:

    >>> !pandoc -t native -f latex test_eqref.tex
    [Para [Math DisplayMath "\\label{blah} hey"]
    ,Para [Str "yo",Space,Str "yo"]]

Notice how the output is missing our `\eqref`.  If we add the `-R` option,
the output will retain all unmatched latex symbols (i.e. `\maketitle` and `\eqref`):

    >>> !pandoc -R -t native -f latex test_eqref.tex
    [RawBlock (Format "latex") "\\maketitle"
    ,Para [Math DisplayMath "\\label{eq1}\nhey"]
    ,Para [Str "yo",Space,Str "yo",Space,RawInline (Format "latex") "\\eqref{eq1}"]]

We can use a filter to retain only the `\eqref` and re-add the `equation[*]`
environment (MathJax can use it via the AMS extensions):

    >>> !pandoc -R -f latex -t json test_eqref.tex | python pandoc_eqref_filter.py | pandoc -f json -t native
    [Null
    ,Para [Math DisplayMath "\\begin{equation}\n\\label{eq1}\nhey\n\\end{equation}"]
    ,Para [Str "yo",Space,Str "yo",Space,RawInline (Format "latex") "\\eqref{eq1}"]]

    >>> !pandoc -R -f latex -t json test_eqref.tex | python pandoc_eqref_filter.py | pandoc -f json -t markdown
    $$\begin{equation}
    \label{eq1}
    hey
    \end{equation}$$

    yo yo \eqref{eq1}


Here is an example with a figure:

    >>> test_file_fig = r'''
    ... [image]: figures/posterior_graph_manipulation_mh_samples_1.pdf \"image title\"
    ...
    ... \\\begin{figure}
    ...     \\\centering
    ...     {\\\includegraphics[width=2.5in]{some_figure.png}}
    ...     \\\caption{Comparing Dq from different p-model}
    ... \\\end{figure}
    ... '''
    >>> !echo "$test_file_fig" >! test_fig.md

    >>> !pandoc -R -f markdown -t json test_fig.md
    | python pandoc_eqref_filter.py

    >>> !pandoc -R -f latex -t json posterior_graph_manipulation.tex | python pandoc_eqref_filter.py
    | pandoc -f json -t markdown

If you want to debug the filter, add a line like this:

.. code:
    import remote_pdb; remote_pdb.RemotePdb('127.0.0.1', 4444).set_trace()

and connect with something like `nc` or `telnet`.

"""

from pandocfilters import toJSONFilter, RawInline, Math

preserved_tex = ['\\eqref', '\\ref', '\\includegraphics']


def filter_eqref(key, value, oformat, meta):
    r""" Keeps unmatched `\eqref`, drops the rest, and wraps
    equation blocks with `equation[*]` environment depending on whether
    or not their body contains a `\label`.
    """
    if key == 'RawInline' and value[0] == 'latex' and \
            any(c_ in value[1] for c_ in preserved_tex):
        return [RawInline('latex', value[1])]
    elif key == "Math" and value[0]['t'] == "DisplayMath":
        star = '*'
        if '\\label' in value[1]:
            star = ''
        wrapped_value = ("\\begin{{equation{}}}\n"
                         "{}\n\\end{{equation{}}}").format(
                             star, value[1], star)
        return Math(value[0], wrapped_value)
    elif "Raw" in key:
        return []


if __name__ == "__main__":
    toJSONFilter(filter_eqref)
