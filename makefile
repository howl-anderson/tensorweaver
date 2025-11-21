format:
	uv run black src/tensorweaver/
	uv run black tests/

test:
	uv run pytest tests/

test-coverage:
	uv run pytest --cov=tensorweaver tests/ --cov-report=html

package:
	uv build

publish:
	# Publishing is now handled by GitHub Actions when you push a tag
	# Example: git tag v0.1.0 && git push origin v0.1.0