import secrets
from functools import wraps
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, g
from werkzeug.security import check_password_hash
from database.db import get_db
from database.models import Admin

auth_bp = Blueprint('auth', __name__)

def generate_csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_hex(16)
    return session['_csrf_token']

def validate_csrf(token):
    return token and token == session.get('_csrf_token')

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('admin_logged_in'):
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('admin_logged_in'):
        return redirect(url_for('dashboard.index'))
        
    csrf_token = generate_csrf_token()
    
    if request.method == 'POST':
        submitted_token = request.form.get('csrf_token')
        if not validate_csrf(submitted_token):
            flash('Invalid session token. Please try again.', 'danger')
            return render_template('login.html', csrf_token=csrf_token)
            
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        if not username or not password:
            flash('Username and password are required.', 'danger')
            return render_template('login.html', csrf_token=csrf_token)
            
        with get_db() as db:
            admin = db.query(Admin).filter_by(username=username).first()
            if admin and check_password_hash(admin.password_hash, password):
                session['admin_logged_in'] = True
                session['admin_id'] = admin.id
                session['admin_username'] = admin.username
                session.permanent = True
                flash(f'Welcome back, {admin.username}!', 'success')
                
                next_url = request.args.get('next')
                if next_url and next_url.startswith('/'):
                    return redirect(next_url)
                return redirect(url_for('dashboard.index'))
            else:
                flash('Invalid username or password.', 'danger')
                
    return render_template('login.html', csrf_token=csrf_token)

@auth_bp.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out securely.', 'info')
    return redirect(url_for('auth.login'))

