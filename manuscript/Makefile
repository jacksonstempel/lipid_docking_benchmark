TEX := manuscript.tex
NAME := manuscript
BUILD_DIR := build
DIST_DIR := dist
SI_TEX := supporting_information.tex
SI_NAME := supporting_information
SI_BUILD_DIR := build_si
SI_DIST_DIR := dist_si

.PHONY: all clean

all: $(DIST_DIR)/$(NAME).pdf $(SI_DIST_DIR)/$(SI_NAME).pdf

FIGURES := $(wildcard figures/*.pdf) $(wildcard figures/*.png)

TEX_ENGINE ?= auto

ifeq ($(TEX_ENGINE),auto)
  ifneq ($(shell command -v tectonic 2>/dev/null),)
    TEX_ENGINE := tectonic
  else
    TEX_ENGINE := pdflatex
  endif
endif

ifeq ($(TEX_ENGINE),tectonic)
$(DIST_DIR)/$(NAME).pdf: $(TEX) references.bib $(FIGURES)
	mkdir -p $(BUILD_DIR) $(DIST_DIR)
	tectonic -X compile --keep-logs --keep-intermediates \
		--outdir $(BUILD_DIR) $(TEX)
	cp $(BUILD_DIR)/$(NAME).pdf $(DIST_DIR)/$(NAME).pdf

$(SI_DIST_DIR)/$(SI_NAME).pdf: $(SI_TEX) references.bib
	mkdir -p $(SI_BUILD_DIR) $(SI_DIST_DIR)
	tectonic -X compile --keep-logs --keep-intermediates \
		--outdir $(SI_BUILD_DIR) $(SI_TEX)
	cp $(SI_BUILD_DIR)/$(SI_NAME).pdf $(SI_DIST_DIR)/$(SI_NAME).pdf
else
$(DIST_DIR)/$(NAME).pdf: $(TEX) references.bib $(FIGURES)
	mkdir -p $(BUILD_DIR) $(DIST_DIR)
	pdflatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(BUILD_DIR) $(TEX)
	bibtex $(BUILD_DIR)/$(NAME)
	pdflatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(BUILD_DIR) $(TEX)
	pdflatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(BUILD_DIR) $(TEX)
	cp $(BUILD_DIR)/$(NAME).pdf $(DIST_DIR)/$(NAME).pdf

$(SI_DIST_DIR)/$(SI_NAME).pdf: $(SI_TEX) references.bib
	mkdir -p $(SI_BUILD_DIR) $(SI_DIST_DIR)
	pdflatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(SI_BUILD_DIR) $(SI_TEX)
	bibtex $(SI_BUILD_DIR)/$(SI_NAME)
	pdflatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(SI_BUILD_DIR) $(SI_TEX)
	pdflatex -interaction=nonstopmode -halt-on-error \
		-output-directory=$(SI_BUILD_DIR) $(SI_TEX)
	cp $(SI_BUILD_DIR)/$(SI_NAME).pdf $(SI_DIST_DIR)/$(SI_NAME).pdf
endif

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR) $(SI_BUILD_DIR) $(SI_DIST_DIR)
