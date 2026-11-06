import pytest
from app import create_app
from config import Config
from database.db import init_db, seed_admin, seed_records_from_csv

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret-key'
    with app.test_client() as client:
        with app.app_context():
            init_db()
            seed_admin()
            seed_records_from_csv()
        yield client

def test_unauthenticated_redirect(client):
    response = client.get('/dashboard')
    assert response.status_code == 302
    assert '/login' in response.headers['Location']

def test_login_flow(client):
    # Fetch login page
    res_get = client.get('/login')
    assert res_get.status_code == 200
    assert b"Administrator Sign In" in res_get.data

    # Obtain session CSRF token
    with client.session_transaction() as sess:
        token = sess.get('_csrf_token')

    # Post invalid password
    res_bad = client.post('/login', data={
        'username': 'admin',
        'password': 'wrongpassword',
        'csrf_token': token
    }, follow_redirects=True)
    assert b"Invalid username or password" in res_bad.data

    # Post valid credentials
    res_good = client.post('/login', data={
        'username': 'admin',
        'password': Config.ADMIN_PASSWORD,
        'csrf_token': token
    }, follow_redirects=True)
    assert res_good.status_code == 200
    assert b"Malaria Epidemiological Dashboard" in res_good.data

def test_authenticated_routes(client):
    # Login as admin
    with client.session_transaction() as sess:
        sess['admin_logged_in'] = True
        sess['admin_username'] = 'admin'

    # 1. Dashboard
    r_dash = client.get('/dashboard')
    assert r_dash.status_code == 200
    assert b"Total Historical Cases" in r_dash.data

    # 2. Dataset
    r_data = client.get('/dataset')
    assert r_data.status_code == 200
    assert b"Automated Data Preprocessing" in r_data.data

    # 3. Analytics
    r_ana = client.get('/analytics')
    assert r_ana.status_code == 200
    assert b"Seasonal Malaria Transmission Patterns" in r_ana.data

    # 4. Predict
    r_pred = client.get('/predict')
    assert r_pred.status_code == 200
    assert b"Forecast Parameters" in r_pred.data

    # 5. Evaluation
    r_eval = client.get('/evaluation')
    assert r_eval.status_code == 200
    assert b"Model Evaluation" in r_eval.data
    assert b"Mean Absolute Error" in r_eval.data

    # 6. History
    r_hist = client.get('/history')
    assert r_hist.status_code == 200
    assert b"Prediction Audit" in r_hist.data

    # 7. APIs
    r_api1 = client.get('/api/dashboard/charts')
    assert r_api1.status_code == 200
    assert 'monthly_trend' in r_api1.json

    r_api2 = client.get('/api/analytics/data')
    assert r_api2.status_code == 200
    assert 'seasonal_patterns' in r_api2.json

    r_api3 = client.get('/api/predict/eligible/Kwara')
    assert r_api3.status_code == 200
    assert 'next_year' in r_api3.json

