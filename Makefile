install:
	pip install uv &&\
	uv sync

test:
	uv run python -m pytest ./tests -vv  --cov=mylib --cov=api --cov=cli 

format:	
	uv run black mylib/*.py api/*.py cli/*.py

lint:
	uv run pylint --disable=R,C --ignore-patterns=test_.*\.py mylib/*.py api/*.py cli/*.py

refactor: format lint
		
all: install format lint test
