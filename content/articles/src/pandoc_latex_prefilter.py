#!/usr/bin/env python
import sys
import re
from copy import copy

import json
import pypandoc

# See https://hackage.haskell.org/package/pandoc-1.17.2
# and https://hackage.haskell.org/package/pandoc-types-1.16.1.1 for
# definitions.
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
env_pattern = re.compile(
    r'\\begin\{(\w+)\}(\[.+\])?(.*)\\end\{\1\}', re.S)
env_label_pattern = re.compile(r'(\s*?\\label\{(\w*?:?\w+)\})')

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
    #     value, meta, args, kwargs))

    if key == 'RawInline' and value[0] == 'latex' and \
            any(c_ in value[1] for c_ in preserved_tex):

        new_value = graphics_pattern.sub(r'{attach}/articles/figures/\2.png',
                                         value[1])
        return Math({'t': 'InlineMath', 'c': []}, new_value)

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
                         "{}\n"
                         "\\end{{equation{}}}").format(
                             star, value[1], star)
        return Math(value[0], wrapped_value)

    if key == 'RawBlock' and value[0] == 'latex':

        env_info = env_pattern.search(value[1])
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

            label_info = env_label_pattern.search(env_body)
            env_label = ""
            label_div = None
            if label_info is not None:
                env_label = label_info.group(2)

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
                env_body = env_body.replace(label_info.group(1), '')


            # type Attr = (String, [String], [(String, String)])
            # Attributes: identifier, classes, key-value pairs
            div_attr = [env_label, [env_name], [['markdown', ''],
                                                ["env-number", str(env_num)],
                                                ['title-name', env_title]
                                                ]]

            # TODO: Should we evaluate nested environments?
            env_body = pypandoc.convert_text(env_body, 'json',
                                             format='latex',
                                             filters=['./pandoc_latex_prefilter.py']
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
