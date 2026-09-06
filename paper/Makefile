# OpenPSN paper — LaTeX source
# Build with tectonic (recommended, self-contained package resolution):
#   tectonic main.tex
# or with a full TeX install:
#   pdflatex main.tex && bibtex not needed (inline thebibliography) && pdflatex main.tex
.PHONY: all pdf clean

PDFLATEX ?= pdflatex
TECTONIC ?= tectonic

all: pdf

pdf: main.tex appendix.tex
	@command -v $(TECTONIC) >/dev/null 2>&1 && $(TECTONIC) main.tex \
	  || $(PDFLATEX) -interaction=nonstopmode main.tex
	@echo " -> main.pdf"

clean:
	rm -f main.pdf main.aux main.log main.out main.synctex.gz *.lof *.lot
