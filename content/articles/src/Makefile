.PHONY: all md clean clean-pweave clean-md clean-pdf clean-tex
#.PRECIOUS: %.tex

PNW_FILES = $(wildcard *.pnw) $(wildcard *.Pnw)
TEXW_FILES = $(wildcard *.texw)
MD_FILES = $(addsuffix .md, $(basename $(PNW_FILES) $(TEXW_FILES)))
TEX_FILES = $(addsuffix .tex, $(basename $(PNW_FILES) $(TEXW_FILES)))
PDF_FILES	= $(TEX_FILES:%.tex=%.pdf)

SRC_DIR = $(abspath ./)
MD_DEST_DIR = $(abspath ../)
PDF_DEST_DIR = $(abspath ./output)

PDFLATEX = "pdflatex -shell-escape -file-line-error -interaction=nonstopmode -synctex=1"

PANDOC_MD_OUT_EXTS = +markdown_in_html_blocks+mmd_title_block+tex_math_dollars+tex_math_double_backslash+implicit_figures+citations+yaml_metadata_block+link_attributes+old_dashes+raw_html+raw_tex

# Add citations with: +citations

# FYI: pandoc's -R option will stop the reader from discarding things like \eqref from
# tex input.
PANDOC_MD_OPTS = -s --mathjax
PANDOC_MD_OPTS += --metadata=figure_dir={attach}/articles/figures/
PANDOC_MD_OPTS += --metadata=figure_ext=png

# Add citations with extra options like these:
# 	--biblio file.bib --csl chicago-author-date.csl
# and possibly this:
# 	--filter pandoc-citeproc

all: md

md: ${MD_FILES}

%.md: %.pnw
	Pweave -c -F $(MD_DEST_DIR)/figures -f pandoc -o $(MD_DEST_DIR)/$@ $<

%.md: %.tex
	pandoc $(PANDOC_MD_OPTS) -f latex -t json --filter pandoc-citeproc $< |\
		PynowebFilter | pandoc $(PANDOC_MD_OPTS) -f json \
		-t markdown_github$(PANDOC_MD_OUT_EXTS) \
		-o $(MD_DEST_DIR)/$@

%.tex: %.texw
	PynowebWeave -c -d -F $(MD_DEST_DIR)/figures -o $(MD_DEST_DIR)/$@ $<

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
	@cd $(PDF_DEST_DIR) && latexmk -f -C -bibtex ${PDF_FILES}

clean-tex:
	-$(RM) $(addprefix $(SRC_DIR)/, ${TEX_FILES})

clean-md:
	-$(RM) $(addprefix $(MD_DEST_DIR)/, ${MD_FILES})

clean-pweave:
	-$(RM) -rf cache
	-$(RM) -rf figures
	-$(RM) -rf $(MD_DEST_DIR)/figures
