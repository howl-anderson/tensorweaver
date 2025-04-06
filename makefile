format:
	poetry run black src/tensorweaver/
	poetry run black tests/

test:
	poetry run pytest tests/

test-coverage:
	poetry run pytest --cov=tensorweaver tests/ --cov-report=html

package:
	poetry build

publish:
	# poetry config pypi-token.pypi <token>
	poetry publish --build

export-requirements:
	poetry self add poetry-plugin-export || true
	poetry export -f requirements.txt --output requirements.txt --without-hashes