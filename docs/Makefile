# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
HOST_PREFIX     ?=
SPHINXOPTS    	?=
SOURCEDIR     	= src
BUILDDIR      	= _build
MODULEDIR		= ../typinox
SPHINXBUILD   	?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
AUTOBUILDIGNORE = --re-ignore ".*autoapi.*"
AUTOBUILDWATCH  = --watch "$(MODULEDIR)"
PREVIEWHOST		?= 0.0.0.0
PREVIEWPORT		?= 18080

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

serve:
	@echo visit "$(HOST_PREFIX)$(PREVIEWPORT)/"
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" --port $(PREVIEWPORT) --host $(PREVIEWHOST) $(AUTOBUILDIGNORE) $(AUTOBUILDWATCH) $(SPHINXOPTS) $(O)
