#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

AUTHOR = u'Brandon T. Willard'
SITENAME = u'Brandon T. Willard'
SITEURL = ''

PATH = 'content'
STATIC_PATHS = ['images','theme','extra/custom.css']

TIMEZONE = 'America/Chicago'

DEFAULT_LANG = u'en'

THEME = "./pelican-themes/pelican-bootstrap3"

#CUSTOM_CSS = 'static/custom.css'
#
## Tell Pelican to change the path to 'static/custom.css' in the output dir
#EXTRA_PATH_METADATA = {
#    'extra/custom.css': {'path': 'static/custom.css'}
#}


# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
#LINKS = (('You can modify those links in your config file', '#'),)

GITHUB_URL = 'https://github.com/brandonwillard'
# Social widget
SOCIAL = (('linkedin', 'http://linkedin.com/pub/brandon-willard/10/bb4/468/'),
          ('google scholar', 'https://scholar.google.com/citations?user=g0oUxG4AAAAJ&hl=enhttps://scholar.google.com/citations?user=g0oUxG4AAAAJ&hl=en'),
          ('google+', 'https://plus.google.com/+brandonwillard'),
          ('bitbucket', 'https://bitbucket.org/brandonwillard'),
          ('gitHub', 'https://github.com/brandonwillard')
          ,)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
#RELATIVE_URLS = True
