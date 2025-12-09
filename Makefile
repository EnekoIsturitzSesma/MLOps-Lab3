install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest ./tests -vv  --cov=mylib --cov=api --cov=cli --ignore=mylib/serialize_best_model.py --ignore=mylib/inference.py --ignore=mylib/train.py 

format:	
	uv run black mylib/*.py api/*.py cli/*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py mylib/*.py api/*.py cli/*.py

refactor: format lint
		
all: install format lint test
