.PHONY: all md clean clean-pweave clean-md
#.PRECIOUS: %.tex

PNW_FILES = $(wildcard *.pnw) $(wildcard *.Pnw)
TEXW_FILES = $(wildcard *.texw)
MD_FILES = $(addsuffix .md, $(basename $(PNW_FILES) $(TEXW_FILES)))
TEX_FILES = $(addsuffix .tex, $(basename $(PNW_FILES) $(TEXW_FILES)))
PDF_FILES	= $(TEX_FILES:%.tex=%.pdf)

PDFLATEX = "pdflatex -shell-escape -file-line-error -interaction=nonstopmode -synctex=1"
 
#HTML_DEPS = $(wildcard _output.yaml) 
#ifneq ($(wildcard include/before_body.html),) 
#PANDOC_OPTIONS += -H include/before_body.html 
#HTML_DEPS += include/before_body.html
#endif

#ifneq ($(wildcard include/after_body.html),) 
#PANDOC_OPTIONS += -A include/after_body.html 
#HTML_DEPS += include/after_body.html
#endif

SRC_DIR = $(realpath $(dir $<))
MD_DEST_DIR = $(abspath ../)
PDF_DEST_DIR = $(abspath ./output)

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
	Pweave -c -F $(MD_DEST_DIR)/figures -f pandoc -o $(MD_DEST_DIR)/$@ $< 

%.md: %.tex
	# FYI: This approach doesn't require the filter to be executable and
	# have a `#!/usr/bin/python` line.
	pandoc $(PANDOC_MD_OPTS) -f latex -t json $< |\
		python pandoc_latex_prefilter.py |\
		pandoc $(PANDOC_MD_OPTS) -f json \
		-t markdown_github$(PANDOC_MD_OUT_EXTS)+raw_html \
		-o $(MD_DEST_DIR)/$@  

%.tex: %.texw
	python pweave_custom.py -c -d -F $(MD_DEST_DIR)/figures $<

#
# the *_line variables will (hopefully) stop unnecessary linebreaks in
# the pdflatex output.  such linebreaks can break error message parsing.
#
%.pdf: %.tex
	@mkdir -p $(PDF_DEST_DIR)
	@cd $(PDF_DEST_DIR)
	@max_print_line=1000 error_line=254 half_error_line=238 openout_any=a\
  							 latexmk -pdf -bibtex -pdflatex=$(PDFLATEX)\
		-jobname=$(PDF_DEST_DIR)/$(<:%.tex=%)\
		$(realpath $<)

clean: clean-tex clean-md clean-pweave clean-pdf

clean-pdf:
	@cd $(PDF_DEST_DIR) && latexmk -C -bibtex ${PDF_FILES}

clean-tex:
	-$(RM) $(addprefix $(PDF_DEST_DIR)/, ${TEX_FILES})

clean-md:
	-$(RM) $(addprefix $(MD_DEST_DIR)/, ${MD_FILES})

clean-pweave:
	-$(RM) -rf cache
	-$(RM) -rf figures
	-$(RM) -rf $(MD_DEST_DIR)/figures
