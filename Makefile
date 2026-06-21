.PHONY: install train test serve dashboard docker-up docker-down lint clean

install:
	pip install -r requirements.txt

train:
	python -m src.models.train

test:
	pytest tests/ -v

serve:
	uvicorn src.api.main:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

mlflow-ui:
	mlflow ui

docker-up:
	docker compose -f docker/docker-compose.yml up --build

docker-down:
	docker compose -f docker/docker-compose.yml down

lint:
	ruff check src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache mlruns
