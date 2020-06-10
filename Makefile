.PHONY: build install test documentation clean static_analyse

build:
	python setup.py build

install:
	python setup.py install

test:
	python setup.py test

documentation:
	pdoc --html --html-dir=doc --overwrite po_nrl

static_analyse:
	bash ./static_analyse.sh

clean:
	rm -f .static_analysis.txt
