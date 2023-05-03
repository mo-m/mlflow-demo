
install:
	pip install -r requirement.txt

train:
	python -m mnist_model.training --option train

search:
	python -m mnist_model.training --option search



