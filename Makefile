.PHONY: help unit dry-run check

help:
	@./run --help

unit:
	@./run test unit

dry-run:
	@./run install --profile deepseek-v4-flash --dry-run
	@./run install --profile glm53-flash --dry-run
	@./run install --profile qwen38-flash --dry-run

check: unit dry-run
