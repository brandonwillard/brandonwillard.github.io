.PHONY: all md clean clean-pweave clean-md
#.PRECIOUS: %.tex

PNW_FILES = $(wildcard *.pnw) $(wildcard *.Pnw)
TEXW_FILES = $(wildcard *.texw)
MD_FILES = $(addsuffix .md, $(basename $(PNW_FILES) $(TEXW_FILES)))

 
#HTML_DEPS = $(wildcard _output.yaml) 
#ifneq ($(wildcard include/before_body.html),) 
#PANDOC_OPTIONS += -H include/before_body.html 
#HTML_DEPS += include/before_body.html
#endif

#ifneq ($(wildcard include/after_body.html),) 
#PANDOC_OPTIONS += -A include/after_body.html 
#HTML_DEPS += include/after_body.html
#endif

SRCDIR = $(realpath $(dir $<))
DEST = $(abspath ../)

PANDOC_MD_OUT_EXTS = +raw_tex+mmd_title_block+tex_math_dollars+tex_math_double_backslash+implicit_figures

# Add citations with: +citations

# FYI: pandoc's -R option will stop the reader from discarding things like \eqref from
# tex input.
PANDOC_MD_OPTS = -R -s --mathjax --old-dashes --self-contained


# Add citations with extra options like these:
# 	--biblio file.bib --csl chicago-author-date.csl 
# and possibly this:
# 	--filter pandoc-citeproc

all: md 

md: ${MD_FILES}

%.md: %.pnw
	Pweave -c -F $(DEST)/figures -f pandoc -o $(DEST)/$@ $< 

%.md: %.tex
	# FYI: This approach doesn't require the filter to be executable and
	# have a `#!/usr/bin/python` line.
	pandoc $(PANDOC_MD_OPTS) -f latex -t json $< |\
		python pandoc_eqref_filter.py |\
		pandoc $(PANDOC_MD_OPTS) -f json \
		-t markdown_github$(PANDOC_MD_OUT_EXTS)+raw_html \
		-o $(DEST)/$@  

%.tex: %.texw
	python pweave_custom.py -c -d -F $(DEST)/figures $<

clean: clean-md clean-pweave

clean-md:
	-$(RM) $(addprefix $(DEST)/, ${MD_FILES})

clean-pweave:
	-$(RM) -rf cache
	-$(RM) -rf figures
	-$(RM) -rf $(DEST)/figures
