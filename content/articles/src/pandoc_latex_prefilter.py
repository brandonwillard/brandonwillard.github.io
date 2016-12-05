r"""
Pandoc filter that doesn't throw away `\eqref{...}` in latex.
Other output formats are unaffected.

Examples
========

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
    ... \textit{some text}
    ... \begin{Exa}
    ...     Blah, blah, blah
    ...     \textit{some more text}
    ...     \label{ex:some_example}
    ... \end{Exa}
    ... \begin{Exa}
    ...     Blahhhhhhhhhhh
    ...     \textit{blohhhhhhh}
    ...     \label{ex:some_example_2}
    ... \end{Exa}
    ...
    ... Example~\ref{ex:some_example} is blah.
    ... Example~\ref{ex:some_example_2} is blah.
    ... '''
    >>> with open('test_exa.tex', 'w') as f:
    >>>     f.write(test_file_exa)

The latex environment comes in the form of a `RawBlock` with format "latex".

    >>> !pandoc -R -f latex -t json test_exa.tex
    [{"unMeta":{}},[{"t":"RawBlock","c":["latex","\\begin{Exa}\nBlah, blah, blah\n\\label{ex:some_example}\n\
    \end{Exa}"]}]]

    >>> !pandoc -R -f latex -t json test_exa.tex | python pandoc_latex_prefilter.py

    >>> !pandoc -R -f latex -t json test_exa.tex | python pandoc_latex_prefilter.py\
        | pandoc -R -f json -t markdown
    <div id="ex:some_example" class="example">
    Blah, blah, blah
    </div>

Don't forget to add the `+markdown_in_html_blocks` option when using other [markdown]
output types:
    >>> !pandoc -R -f latex -t json test_exa.tex | python pandoc_latex_prefilter.py\
        | pandoc -R -f json -t markdown_github
    Blah, blah, blah

    >>> !pandoc -R -f latex -t json test_exa.tex | python pandoc_latex_prefilter.py\
        | pandoc -R -f json -t markdown_github+markdown_in_html_blocks
    <div id="ex:some_example" class="example">
    Blah, blah, blah
    </div>

If you want to debug the filter, add a line like this:

.. code:
    import remote_pdb; remote_pdb.RemotePdb('127.0.0.1', 4444).set_trace()

and connect with something like `nc` or `telnet`.

"""

import sys
import re
from copy import copy

import json
import pypandoc

from pandocfilters import (
    toJSONFilter,
    Math,
    Image,
    Div,
    Str,
    Plain,
    RawInline,
    RawBlock)

graphics_pattern = re.compile(
    r"includegraphics(?:\[.+\])?\{(.*?(\w+)(\.\w*))\}")
image_pattern = re.compile(r"(.*?(\w+)(\.\w*))$")
label_pattern = re.compile(r'\s*\\label\{(\w*?:?\w+)\}')
environment_pattern = re.compile(
    r'\\begin\{(\w+)\}(\[.+\])?(.*)\\end\{\1\}', re.S)

# TODO: Should get from `meta` parameter in the filter?
environment_conversions = {'Exa': 'example'}

environment_counters = {}

preserved_tex = ['\\eqref', '\\ref', '\\includegraphics']


def latex_prefilter(key, value, oformat, meta, *args, **kwargs):
    r""" A prefilter that adds more latex capabilities to Pandoc's tex to
    markdown features.

    Currently implemented:
        * Keeps unmatched `\eqref` (drops the rest)
        * Wraps equation blocks with `equation[*]` environment depending on
          whether or not their body contains a `\label`
        * Converts custom environments to div objects

    Set the variables `preserved_tex` and `environment_conversions` to
    allow more raw latex commands and to convert latex environment names
    to CSS class names, respectively.

    """
    # sys.stderr.write("Filter args: {}, {}, {}, {}\n".format(
    #     oformat, meta, args, kwargs))

    if key == 'RawInline' and value[0] == 'latex' and \
            any(c_ in value[1] for c_ in preserved_tex):

        new_value = graphics_pattern.sub(r'{attach}/articles/figures/\2.png',
                                         value[1])
        return RawInline('latex', new_value)

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

    if key == 'RawBlock' and value[0] == 'latex':

        env_info = environment_pattern.search(value[1])
        if env_info is not None:
            env_groups = env_info.groups()
            env_name = env_groups[0]
            env_name = environment_conversions.get(env_name, env_name)
            env_title = env_groups[1]

            if env_title is None:
                env_title = ""

            env_body = env_groups[2]

            env_num = environment_counters.get(env_name, 0)
            env_num += 1
            environment_counters[env_name] = env_num

            label_info = label_pattern.search(env_body)
            env_label = ""
            label_div = None
            if label_info is not None:
                env_label = label_info.group(1)

                hack_div_label = env_label+"_math"
                # XXX: We're hijacking MathJax's numbering system.
                ref_hack = (r'$$\begin{{equation}}'
                            r'\tag{{{}}}'
                            r'\label{{{}}}'
                            r'\end{{equation}}$$'
                            ).format(env_num, env_label)

                label_div = Div([hack_div_label, [],
                                 [#['markdown', ''],
                                  ["style",
                                   "display:none;visibility:hidden"]]],
                                [RawBlock('latex', ref_hack)])

                # Now, remove the latex label string
                env_body = label_pattern.sub(r'', env_body)

            # type Attr = (String, [String], [(String, String)])
            # Attributes: identifier, classes, key-value pairs
            div_attr = [env_label, [env_name], [['markdown', ''],
                                                ["env-number", str(env_num)],
                                                ['title-name', env_title]
                                                ]]

            # TODO: Can/should we evaluate nested environments?
            env_body = pypandoc.convert_text(env_body, 'json',
                                             format='latex',
                                             filters=None
                                             # ['pandoc_latex_prefilter.py']
                                             )

            div_block = json.loads(env_body)[1]

            if label_div is not None:
                div_block = [label_div] + div_block

            return Div(div_attr, div_block)
        else:
            return []

    elif "Raw" in key:
        return []


if __name__ == "__main__":
    toJSONFilter(latex_prefilter)
