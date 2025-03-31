#!/bin/bash

# Build script for compiling the LaTeX document

cd "$(dirname "$0")"

# Compile the document
pdflatex gaugefixer.tex
bibtex gaugefixer
pdflatex gaugefixer.tex
pdflatex gaugefixer.tex

echo "Build complete. Output file: gaugefixer.pdf" 