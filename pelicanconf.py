#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import sys

PLUGIN_PATHS = ["/home/bwillar0/apps/pelican-plugins"]

PLUGINS = ["render_math", "liquid_tags"]

AUTHOR = u'Brandon T. Willard'
SITENAME = u'Brandon T. Willard'
SITEURL = ''

PATH = 'content'
STATIC_PATHS = [
    'images',
    'extra/custom.css',
    'extra/favicon.ico',
    'extra/academicons-1.7.0/css',
    'extra/academicons-1.7.0/fonts',
]
EXTRA_PATH_METADATA = {
    'extra/favicon.ico': {'path': 'favicon.ico'}
}

PAGE_PATHS = ['pages']
ARTICLE_PATHS = ['articles']

TIMEZONE = 'America/Chicago'

DEFAULT_LANG = u'en'

THEME = "/home/bwillar0/apps/pelican-themes/pelican-bootstrap3"

#CUSTOM_CSS = 'static/custom.css'
DISQUS_SITENAME = "brandonwillard-github-io"

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

ABOUT_ME = "applied math/stats person"
AVATAR = '/images/profile-pic.png'
#PROFILE_PICTURE = '/images/profile-pic.png'

# Blogroll
#LINKS = (('You can modify those links in your config file', '#'),)

GITHUB_URL = 'https://github.com/brandonwillard'

# Social widget
SOCIAL = (('linkedin', 'http://linkedin.com/pub/brandon-willard/10/bb4/468/'),
          ('google scholar', 'https://scholar.google.com/citations?user=g0oUxG4AAAAJ&hl=en'),
          ('google+', 'https://plus.google.com/+brandonwillard'),
          ('bitbucket', 'https://bitbucket.org/brandonwillard'),
          ('github', 'https://github.com/brandonwillard')
          ,)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = False

MATH_JAX = {'linebreak_automatic': True,
            'tex_extensions': ['AMSmath.js', 'AMSsymbols.js']
            }
