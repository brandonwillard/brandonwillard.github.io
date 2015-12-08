# Introduction

This is a personal [Pelican](http://docs.getpelican.com/) site.

# Setup

Basically, set up Pelican (using `pelican-quickstart` for instance). 

# Usage

See [this](http://docs.getpelican.com/en/3.6.3/tips.html) for info on pushing to Github pages.
Quickly,
```
$ pelican content -o output -s pelicanconf.py
$ ghp-import output
$ git push git@github.com:brandonwillard/brandonwillard.github.io.git gh-pages:master
```

