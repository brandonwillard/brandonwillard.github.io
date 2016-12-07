# Introduction

This is a personal [Pelican](http://docs.getpelican.com/) site.

# Setup

Basically, set up Pelican (using `pelican-quickstart` for instance). 

This site also uses the plugin [render_math](https://github.com/barrysteyn/pelican_plugin-render_math), so install it (reference the settings file for exactly how).

# Usage

## Preview
First start the server:
```
$ ./develop_server.sh start
```
then check [localhost](http://localhost:8000/).

## IPython

The plugin [Liquid Tags](https://github.com/getpelican/pelican-plugins/tree/master/liquid_tags) works with the [pelican-bootstrap3](https://github.com/DandyDev/pelican-bootstrap3) theme to embed IPython notebooks with lines like
```
{% notebook filename.ipynb %}
```

Amazing!

## Publishing

See [this](http://docs.getpelican.com/en/3.6.3/tips.html#publishing-to-github)
for info on pushing to Github pages.
Quickly,
```
$ pelican content -o output -s publishconf.py
$ ghp-import output
$ git push git@github.com:brandonwillard/brandonwillard.github.io.git gh-pages:master
```
or simply `$ make github`.

## FYI

I've altered the default `pelican-bootstrap3` theme to include the Google Scholar icons
provided by [Academicons](https://jpswalsh.github.io/academicons/).  The changes are in
`pelican-bootstrap3/templates/includes/sidebar.html` within the `SOCIAL` for-loop:
```
{% elif name_sanitized in ['google-scholar'] %}
    {% set iconattributes = '"ai ai-' ~ name_sanitized ~ ' ai-lg"' %}
```
Of course this also means the `fonts` and `css` folders must be accessible and the CSS
stylesheet loaded.
