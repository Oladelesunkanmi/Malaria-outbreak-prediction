import os
import tempfile
import pytest
from config import Config
from database.db import set_engine_and_session, init_db, seed_admin, seed_records_from_csv
from app import create_app

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Session-scoped fixture creating a fresh temporary SQLite database
    for the entire test run, completely isolating tests from malaria_predict.db.
    """
    original_uri = Config.SQLALCHEMY_DATABASE_URI
    fd, temp_db_path = tempfile.mkstemp(suffix="_test_malaria.db")
    os.close(fd)
    
    test_uri = f"sqlite:///{temp_db_path}"
    Config.SQLALCHEMY_DATABASE_URI = test_uri
    set_engine_and_session(test_uri)
    
    init_db()
    seed_admin()
    seed_records_from_csv()
    
    yield temp_db_path
    
    # Teardown: restore original URI and remove temp file
    Config.SQLALCHEMY_DATABASE_URI = original_uri
    set_engine_and_session(original_uri)
    try:
        if os.path.exists(temp_db_path):
            os.remove(temp_db_path)
    except Exception:
        pass

@pytest.fixture
def app():
    """Create Flask test application configured for testing."""
    test_app = create_app()
    test_app.config['TESTING'] = True
    test_app.config['WTF_CSRF_ENABLED'] = False
    return test_app

@pytest.fixture
def client(app):
    """Provide an unauthenticated Flask test client."""
    return test_app_client(app)

def test_app_client(app):
    return app.test_client()

@pytest.fixture
def auth_client(app):
    """Provide an authenticated Flask test client."""
    c = app.test_client()
    with c.session_transaction() as sess:
        sess['admin_logged_in'] = True
        sess['admin_username'] = Config.ADMIN_USERNAME
    return c

