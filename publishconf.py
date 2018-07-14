#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals

import os
import sys

# Hackish way of making additions to pelicanconf.py
sys.path.append(os.curdir)

from pelicanconf import *  # noqa

# This file is only used if you use `make publish` or
# explicitly specify it as your config file.

SITEURL = 'https://brandonwillard.github.io'
RELATIVE_URLS = False

FEED_DOMAIN = SITEURL
FEED_ALL_ATOM = 'feeds/all.atom.xml'
FEED_ALL_RSS = 'feeds/all.rss.xml'
CATEGORY_FEED_ATOM = 'feeds/%s.atom.xml'

DELETE_OUTPUT_DIRECTORY = True
LOAD_CONTENT_CACHE = False

DISQUS_SITENAME = "brandonwillard-github-io"
GOOGLE_ANALYTICS_UNIVERSAL = "UA-91585967-1"
