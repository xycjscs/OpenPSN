# OpenPSN paper — LaTeX source
# Build with tectonic (recommended, self-contained package resolution):
#   tectonic main.tex          (framework paper)
#   tectonic sector_method.tex (companion paper on the finite-sector closure)
# or with a full TeX install:
#   pdflatex main.tex && bibtex not needed (inline thebibliography) && pdflatex main.tex
.PHONY: all pdf companion docx clean

PDFLATEX ?= pdflatex
TECTONIC ?= tectonic
PANDOC ?= pandoc

all: pdf companion

pdf: main.tex appendix.tex tables/study4-checker.tex tables/study4-assembly.tex
	@command -v $(TECTONIC) >/dev/null 2>&1 && $(TECTONIC) main.tex \
	  || $(PDFLATEX) -interaction=nonstopmode main.tex
	@echo " -> main.pdf"

# The companion paper reuses figures/ and the generated tables/, and cites the
# framework paper. Its tables must be regenerated first (see the report scripts
# in ../scripts) because they are checked-out build products, not hand-written.
companion: sector_method.tex tables/study4-checker.tex tables/study4-assembly.tex \
           tables/study4-core-pcm.tex tables/study4-core-time.tex tables/study4-core-best.tex
	@command -v $(TECTONIC) >/dev/null 2>&1 && $(TECTONIC) sector_method.tex \
	  || $(PDFLATEX) -interaction=nonstopmode sector_method.tex
	@echo " -> sector_method.pdf"

docx: main.tex word-math.lua
	$(PANDOC) --from latex --resource-path .:figures --reference-doc main.docx \
	  --lua-filter word-math.lua main.tex -o main.generated.docx
	mv main.generated.docx main.docx

clean:
	rm -f main.pdf main.aux main.log main.out main.synctex.gz *.lof *.lot
	rm -f sector_method.pdf sector_method.aux sector_method.log sector_method.out
