import sys
import os

import uvicorn

# Add the directory containing the back package to Python path for PyInstaller
if getattr(sys, 'frozen', False):
    # Running as PyInstaller bundle
    bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, bundle_dir)

if __name__ == "__main__":
    """Convenience entry-point to launch the FastAPI backend.

    Run `python -m back.server` (or `python back/server.py`) and then visit
    http://localhost:8000/docs for the interactive Swagger UI or
    http://localhost:8000/redoc for ReDoc. The OpenAPI JSON is available at
    http://localhost:8000/openapi.json.
    """
    # Import here to avoid side-effects if the module is imported elsewhere.
    # Handle both relative and absolute imports for PyInstaller compatibility
    try:
        from .api import app  # noqa: WPS433 – internal import for runtime
    except (ImportError, ValueError):
        try:
            from backend.api import app
        except ImportError:
            # Last resort for PyInstaller
            import importlib.util
            spec = importlib.util.spec_from_file_location("api", os.path.join(os.path.dirname(__file__), "api.py"))
            api_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(api_module)
            app = api_module.app

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False) 