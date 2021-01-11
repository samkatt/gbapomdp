.PHONY: build install test documentation clean static_analyse ctags

build:
	python setup.py build

install:
	python setup.py install

test:
	python setup.py test

documentation:
	pdoc3 --html --html-dir=doc --overwrite po_nrl

static_analyse:
	bash ./static_analyse.sh

ctags:
	ctags -R tests po_nrl

clean:
	rm -f .static_analysis.txt
