#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals


PLUGIN_PATHS = ["./pelican-plugins"]

PLUGINS = ["i18n_subsites", "liquid_tags", "pelican-bibtex", "pandoc_reader"]

JINJA_ENVIRONMENT = {
    "extensions": ["jinja2.ext.i18n"],
}

PUBLICATIONS_SRC = "content/pages/publications.bib"

AUTHOR = "Brandon T. Willard"
SITENAME = "Brandon T. Willard"
RELATIVE_URLS = True
SITEURL = ""

CC_LICENSE = "CC-BY-NC"

PATH = "content"
STATIC_PATHS = [
    "images",
    "extra/favicon.ico",
    "extra/custom.css",
    "articles/figures",
]
EXTRA_PATH_METADATA = {"extra/favicon.ico": {"path": "favicon.ico"}}

PAGE_PATHS = ["pages"]
ARTICLE_PATHS = ["articles"]
ARTICLE_EXCLUDES = ["articles/src"]
SUMMARY_MAX_LENGTH = 0

TIMEZONE = "America/Chicago"

DEFAULT_DATE = "fs"
DEFAULT_LANG = "en"

THEME = "pelican-themes/pelican-bootstrap3"
BOOTSTRAP_THEME = "readable"
PYGMENTS_STYLE = "vim"

CUSTOM_CSS = "extra/custom.css"

DISQUS_SITENAME = "brandonwillard-github-io"

# Feed generation is usually not desired when developing
# FEED_ALL_ATOM = None
# FEED_ALL_ATOM = 'feeds/all.atom.xml'
# FEED_ALL_RSS = 'feeds/all.rss.xml'
# CATEGORY_FEED_ATOM = None
# TRANSLATION_FEED_ATOM = None
# AUTHOR_FEED_ATOM = None
# AUTHOR_FEED_RSS = None

ABOUT_ME = "applied math/stats person"
AVATAR = "/images/profile-pic.png"
# PROFILE_PICTURE = '/images/profile-pic.png'

# Blogroll
# LINKS = (('You can modify those links in your config file', '#'),)

GITHUB_URL = "https://github.com/brandonwillard"

# Social widget
SOCIAL = (
    ("github", "https://github.com/brandonwillard"),
    ("google scholar", "https://scholar.google.com/citations?user=g0oUxG4AAAAJ&hl=en"),
    ("linkedin", "https://linkedin.com/pub/brandon-willard/10/bb4/468/"),
    ("bitbucket", "https://bitbucket.io/brandonwillard"),
)

DISPLAY_TAGS_ON_SIDEBAR = True
DISPLAY_TAGS_INLINE = True
SHOW_DATE_MODIFIED = True

DIRECT_TEMPLATES = [
    "index",
    "archives",  # 'publications'
]
EXTRA_TEMPLATES_PATHS = ["custom"]

PAGES_SORT_ATTRIBUTE = "date"
ARTICLE_ORDER_BY = "reversed-date"

DEFAULT_PAGINATION = 10

PANDOC_BIBHEADER = "References"
PANDOC_BIBDIR = "./content/articles/src"
PANDOC_ARGS = [
    "-s",
    "--mathjax",
    "--section-divs",
    "--highlight-style=pygments",
    "--include-after-body=./content/articles/src/after_body.html",
    "--template=pelican_template.html",
]
PANDOC_EXTENSIONS = [
    "+old_dashes",
    "+yaml_metadata_block",
    "+raw_tex",
    "+auto_identifiers",
    "+tex_math_single_backslash",
    "+link_attributes",
    "+fenced_code_attributes",
]
PANDOC_FILTERS = ["pandoc-citeproc"]

DELETE_OUTPUT_DIRECTORY = True
