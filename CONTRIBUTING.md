# Contributing

## Install dev dependencies

If using pip, you must do this explicitly with:

```
pip install -e ".[dev]"
```

## Before commiting code / making a PR

Run the following...

- Ruff - linting and formatting
    ```bash
    uv run ruff check --fix .
    ```

- ty - type hint checking
    ```bash
    uv run ty check . 
    ```

- isort - order imports
    ```bash
    uv run isort .
    ```

> [!NOTE]  
> ToDo: Add these as pre-commit hooks