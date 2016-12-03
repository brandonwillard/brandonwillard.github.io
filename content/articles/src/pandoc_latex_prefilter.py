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

    >>> !pandoc -R -f latex -t json test_eqref.tex | python pandoc_latex_prefilter.py | pandoc -f json -t native
    [Null
    ,Para [Math DisplayMath "\\begin{equation}\n\\label{eq1}\nhey\n\\end{equation}"]
    ,Para [Str "yo",Space,Str "yo",Space,RawInline (Format "latex") "\\eqref{eq1}"]]

    >>> !pandoc -R -f latex -t json test_eqref.tex | python pandoc_latex_prefilter.py | pandoc -f json -t markdown
    $$\begin{equation}
    \label{eq1}
    hey
    \end{equation}$$

    yo yo \eqref{eq1}


Here is an example with a figure:

    >>> test_file_fig = r'''
    ...
    ... \begin{figure}[htpb]
    ... \center
    ... \includegraphics{/home/bwillar0/projects/websites/brandonwillard.github.io/content/articles/src/../figures/regarding_sample_estimates_temp_ppc_plot_1.png}
    ... \caption{Posterior samples}
    ... \label{fig:temp_ppc_plot}
    ... \end{figure}
    ...
    ... \begin{figure}
    ...     \centering
    ...     {\includegraphics[width=2.5in]{some_figure.png}}
    ...     \\caption{Comparing Dq from different p-model}
    ... \end{figure}
    ... '''
    >>> with open('test_fig.tex', 'w') as f:
    >>>     f.write(test_file_fig)

    >>> !pandoc -R -f latex -t json test_fig.tex

    >>> !pandoc -R -f latex -t json test_fig.tex | python pandoc_latex_prefilter.py

Next, let's see how latex environments can be converted.
First, let's get an example of what we're aiming for:
    >>> test_file_exa_html = r'''
    ...
    ... <div class="theorem" style="content:'Theorem (Prime numbers)';">
    ... All odd numbers are prime.
    ... </div>
    ... '''
    >>> with open('test_exa.html', 'w') as f:
    >>>     f.write(test_file_exa_html)

    >>> !pandoc -R -f html -t json test_exa.html
    [{"unMeta":{}},[{"t":"Div","c":[["",["theorem"],[["style","content:'Theorem (Prime numbers)';"]]],[{"t":"Plain","c":[
    {"t":"Str","c":"All"},{"t":"Space","c":[]},{"t":"Str","c":"odd"},{"t":"Space","c":[]},{"t":"Str","c":"numbers"},{"t":
    "Space","c":[]},{"t":"Str","c":"are"},{"t":"Space","c":[]},{"t":"Str","c":"prime."}]}]]}]]

So we need to create a `Div` object like above.

    >>> test_file_exa = r'''
    ... \begin{Exa}
    ... Blah, blah, blah
    ... \label{ex:some_example}
    ... \end{Exa}
    ...
    ... '''
    >>> with open('test_exa.tex', 'w') as f:
    >>>     f.write(test_file_exa)

The latex environment comes in the form of a `RawBlock` with format "latex".

    >>> !pandoc -R -f latex -t json test_exa.tex
    [{"unMeta":{}},[{"t":"RawBlock","c":["latex","\\begin{Exa}\nBlah, blah, blah\n\\label{ex:some_example}\n\
    \end{Exa}"]}]]

    >>> !pandoc -R -f latex -t json test_exa.tex | python pandoc_latex_prefilter.py
    [{"unMeta": {}}, [{"c": [["", ["example"], [["style", "content:'None';"]]], [{"c": ["html", "\nBlah, blah
    , blah\n\\label{ex:some_example}\n"], "t": "RawBlock"}]], "t": "Div"}]]

    >>> !pandoc -R -f latex -t json test_exa.tex | python pandoc_latex_prefilter.py\
        | pandoc -R -f json -t markdown
    <div class="example" style="content:'None';">
    Blah, blah, blah
    \label{ex:some_example}
    </div>

If you want to debug the filter, add a line like this:

.. code:
    import remote_pdb; remote_pdb.RemotePdb('127.0.0.1', 4444).set_trace()

and connect with something like `nc` or `telnet`.

"""

from pandocfilters import (
    toJSONFilter,
    RawInline,
    Math,
    Image,
    Div,
    Para,
    Str,
    RawBlock)
import re
from copy import copy

graphics_pattern = re.compile(
    r"includegraphics(?:\[.+\])?\{(.*?(\w+)(\.\w*))\}")
image_pattern = re.compile(r"(.*?(\w+)(\.\w*))$")
label_pattern = re.compile(r'\\label\{(\w*?:?\w+)\}')
environment_pattern = re.compile(
    r'\\begin\{(\w+)\}(\[.+\])?(.*)\\end\{\1\}', re.S)

# TODO: Should get from `meta` parameter in the filter?
environment_conversions = {'Exa': 'example'}

preserved_tex = ['\\eqref', '\\ref', '\\includegraphics']


def latex_prefilter(key, value, oformat, meta):
    r"""
        * Keeps unmatched `\eqref`, drops the rest
        * Wraps equation blocks with `equation[*]` environment depending on whether
          or not their body contains a `\label`.
        * Converts custom environments to div objects.
    """
    if key == 'RawInline' and value[0] == 'latex' and \
            any(c_ in value[1] for c_ in preserved_tex):

        new_value = graphics_pattern.sub(r'{attach}/articles/figures/\2.png',
                                         value[1])
        return RawInline('latex', new_value)

    if key == 'RawBlock' and value[0] == 'latex':

        env_info = environment_pattern.search(value[1])
        if env_info is not None:
            env_groups = env_info.groups()
            env_name = env_groups[0]
            env_name = environment_conversions.get(env_name, env_name)
            env_title = env_groups[1]
            env_body = env_groups[2]

            # TODO: Use labels.
            # env_body = label_pattern.sub(r'\1', env_body)

            new_value = [
                ["", [env_name], [["style", "content:'{}';".format(env_title)]]],
                [RawBlock('html', env_body)]
                # [Para([Str(env_body)])]
            ]

            return Div(*new_value)

    elif key == "Image":

        # TODO: Find and use labels.
        new_value = copy(value[2])
        new_value[0] = image_pattern.sub(r'{attach}/articles/figures/\2.png',
                                         new_value[0])
        return Image(value[0], value[1], new_value)

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
    toJSONFilter(latex_prefilter)
