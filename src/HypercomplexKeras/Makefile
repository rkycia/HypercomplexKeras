

PYTHON=python3

run: Algebra.py Hyperdense.py Convolutional.py
	chmod +x $<
	$(PYTHON) Algebra.py
	$(PYTHON) Hyperdense.py
	$(PYTHON) Convolutional.py
	$(PYTHON) HyperdenseTorch.py
	$(PYTHON) ExperimentalHyperConvolutionalTorch.py

generate_doc: doc_generator.py
	$(PYTHON) doc_generator.py
	-mkdir doc
	mv *.html ./doc
	firefox ./doc/Hyperdense.html

clean_doc:
	-rm -r ./doc

clean: clean_doc
	-rm -r __pycache__
