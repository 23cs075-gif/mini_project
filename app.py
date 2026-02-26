import os
from flask import Flask, request, jsonify, session, redirect
from config import Config
from extensions import db
from sqlalchemy import text

# make sure models are imported so SQLAlchemy knows about them
import models
from models import Student, Teacher

# import models module so that SQLAlchemy is aware of all subclasses of
# db.Model before create_all() is called.  the import has no other side
# effects; it merely triggers the class definitions in models.py.
import models


def create_app():
    """Application factory that configures Flask and extensions."""
    # serve static assets (css/js/images, and the existing html files) from
    # the project root so that the original index.html can work without any
    # modification to its relative URLs.
    app = Flask(__name__, static_folder='.', static_url_path='')
    app.config.from_object(Config)
    # set secret key for session
    app.secret_key = app.config.get('SECRET_KEY')

    # initialize extensions with the app
    db.init_app(app)

    # Global error handler to return JSON for API endpoints
    @app.errorhandler(Exception)
    def handle_error(e):
        app.logger.error(f"Unhandled error: {e}")
        if request.path.startswith('/api/'):
            return jsonify(message='server error'), 500
        raise e

    # make sure models are imported after db.bind; this is normally handled
    # by the top-level import above but kept here for clarity.
    # (``models`` already imported at module level.)

    @app.route("/")
    def index():
        # test database connectivity but always return the front page if
        # available. we log the result to the console so developers can see
        # connection problems without having to change the HTML.
        try:
            db.session.execute(text("SELECT 1"))
            app.logger.info("Database connection successful")
        except Exception as e:
            app.logger.warning(f"Database connection failed: {e}")

        # send the existing index.html in the repository root
        return app.send_static_file('index.html')

    # ------------------------------------------------------------------
    # student authentication/api endpoints
    # ------------------------------------------------------------------

    def _current_student():
        sid = session.get('student_id')
        if sid:
            return Student.query.get(sid)
        return None

    @app.route('/api/student/register', methods=['POST'])
    def student_register():
        try:
            data = request.get_json() or {}
            # basic validation
            if not data.get('username') or not data.get('password'):
                return jsonify(message='username and password required'), 400

            # prevent duplicates by username or email or roll_number
            exists = Student.query.filter(
                (Student.username == data.get('username')) |
                (Student.email == data.get('email')) |
                (Student.roll_number == data.get('roll_number'))
            ).first()
            if exists:
                return jsonify(message='user already exists'), 400

            student = Student(
                name=data.get('name'),
                username=data.get('username'),
                email=data.get('email'),
                department=data.get('department'),
                phone=data.get('phone'),
                roll_number=data.get('roll_number'),
                semester=data.get('semester')
            )
            student.set_password(data.get('password'))
            if data.get('photo'):
                student.photo = data.get('photo')
            db.session.add(student)
            db.session.commit()
            return jsonify(message='registered'), 200
        except Exception as e:
            app.logger.error(f"Registration error: {e}")
            return jsonify(message='registration failed'), 500

    @app.route('/api/student/login', methods=['POST'])
    def student_login():
        try:
            data = request.get_json() or {}
            username = data.get('username')
            password = data.get('password')
            student = None
            if username:
                student = Student.query.filter_by(username=username).first()
            if not student and data.get('roll_number'):
                student = Student.query.filter_by(roll_number=data.get('roll_number')).first()
            if not student or not student.check_password(password):
                return jsonify(message='invalid credentials'), 401
            session['student_id'] = student.id
            return jsonify(message='logged in'), 200
        except Exception as e:
            app.logger.error(f"Login error: {e}")
            return jsonify(message='login failed'), 500

    @app.route('/api/student/logout', methods=['POST'])
    def student_logout():
        session.pop('student_id', None)
        return jsonify(message='logged out')

    @app.route('/api/student/me', methods=['GET'])
    def student_me():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            return jsonify(
                id=student.id,
                username=student.username,
                name=student.name,
                email=student.email,
                department=student.department,
                phone=student.phone,
                roll_number=student.roll_number,
                semester=student.semester,
                photo=student.photo,
            )
        except Exception as e:
            app.logger.error(f"Get student info error: {e}")
            return jsonify(message='failed to get student info'), 500

    @app.route('/api/student/update', methods=['POST'])
    def student_update():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            # only update known fields
            for field in ('name', 'email', 'department', 'phone', 'roll_number', 'semester', 'photo', 'username'):
                if field in data:
                    setattr(student, field, data[field])
            db.session.commit()
            return jsonify(message='updated')
        except Exception as e:
            app.logger.error(f"Update student error: {e}")
            return jsonify(message='update failed'), 500

    @app.route('/api/student/change_password', methods=['POST'])
    def student_change_password():
        try:
            student = _current_student()
            if not student:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            current = data.get('current_password')
            new = data.get('new_password')
            if not student.check_password(current):
                return jsonify(message='incorrect current password'), 400
            student.set_password(new)
            db.session.commit()
            return jsonify(message='password changed')
        except Exception as e:
            app.logger.error(f"Change password error: {e}")
            return jsonify(message='password change failed'), 500

    # ------------------------------------------------------------------
    # teacher authentication/api endpoints
    # ------------------------------------------------------------------

    def _current_teacher():
        tid = session.get('teacher_id')
        if tid:
            return Teacher.query.get(tid)
        return None

    @app.route('/api/teacher/register', methods=['POST'])
    def teacher_register():
        try:
            data = request.get_json() or {}
            if not data.get('username') or not data.get('password'):
                return jsonify(message='username and password required'), 400

            exists = Teacher.query.filter(
                (Teacher.username == data.get('username')) |
                (Teacher.email == data.get('email'))
            ).first()
            if exists:
                return jsonify(message='user already exists'), 400

            teacher = Teacher(
                name=data.get('name'),
                username=data.get('username'),
                email=data.get('email'),
                department=data.get('department'),
                phone=data.get('phone')
            )
            teacher.set_password(data.get('password'))
            db.session.add(teacher)
            db.session.commit()
            return jsonify(message='registered'), 200
        except Exception as e:
            app.logger.error(f"Teacher registration error: {e}")
            return jsonify(message='registration failed'), 500

    @app.route('/api/teacher/login', methods=['POST'])
    def teacher_login():
        try:
            data = request.get_json() or {}
            username = data.get('username')
            password = data.get('password')
            teacher = None
            if username:
                teacher = Teacher.query.filter_by(username=username).first()
            if not teacher or not teacher.check_password(password):
                return jsonify(message='invalid credentials'), 401
            session['teacher_id'] = teacher.id
            return jsonify(message='logged in'), 200
        except Exception as e:
            app.logger.error(f"Teacher login error: {e}")
            return jsonify(message='login failed'), 500

    @app.route('/api/teacher/logout', methods=['POST'])
    def teacher_logout():
        session.pop('teacher_id', None)
        return jsonify(message='logged out')

    @app.route('/api/teacher/me', methods=['GET'])
    def teacher_me():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            return jsonify(
                id=teacher.id,
                username=teacher.username,
                name=teacher.name,
                email=teacher.email,
                department=teacher.department,
                phone=teacher.phone
            )
        except Exception as e:
            app.logger.error(f"Get teacher info error: {e}")
            return jsonify(message='failed to get teacher info'), 500

    @app.route('/api/teacher/update', methods=['POST'])
    def teacher_update():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            for field in ('name', 'email', 'department', 'phone', 'username'):
                if field in data:
                    setattr(teacher, field, data[field])
            db.session.commit()
            return jsonify(message='updated')
        except Exception as e:
            app.logger.error(f"Update teacher error: {e}")
            return jsonify(message='update failed'), 500

    @app.route('/api/teacher/change_password', methods=['POST'])
    def teacher_change_password():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401
            data = request.get_json() or {}
            current = data.get('current_password')
            new = data.get('new_password')
            if not teacher.check_password(current):
                return jsonify(message='incorrect current password'), 400
            teacher.set_password(new)
            db.session.commit()
            return jsonify(message='password changed')
        except Exception as e:
            app.logger.error(f"Change teacher password error: {e}")
            return jsonify(message='password change failed'), 500
        # -------------------------------------------------------------
    # GET ALL STUDENTS (for Teacher Dashboard)
    # -------------------------------------------------------------
    @app.route('/api/teacher/students', methods=['GET'])
    def teacher_get_students():
        try:
            teacher = _current_teacher()
            if not teacher:
                return jsonify(message='not authenticated'), 401

            students = Student.query.all()

            result = []
            for s in students:
                result.append({
                    "id": s.id,
                    "name": s.name,
                    "username": s.username,
                    "email": s.email,
                    "department": s.department,
                    "phone": s.phone,
                    "roll_number": s.roll_number,
                    "semester": s.semester
                })

            return jsonify(students=result)

        except Exception as e:
            app.logger.error(f"Get students error: {e}")
            return jsonify(message='failed to fetch students'), 500

    return app


def create_tables(app=None):
    """Create all database tables.

    If an app is not provided, a new one will be created using :func:`create_app`.
    This function should be called from a context where the application
    configuration has been loaded (for example, from a script or a shell).

    The database URI may point to a MySQL server that is unreachable or
    requires credentials we don't have; in that case we automatically fall
    back to a local SQLite file so the rest of the application can still run.
    """
    if app is None:
        app = create_app()

    # flask-sqlalchemy needs an application context for create_all
    with app.app_context():
        try:
            # first attempt to create tables using whatever URI is configured
            db.create_all()
        except Exception as exc:
            # if there was any problem (e.g. access denied) log it and
            # switch to sqlite fallback before trying once more.
            app.logger.warning(f"primary create_all failed: {exc}")

            fallback = "sqlite:///fallback.db"
            app.logger.info(f"switching to fallback database {fallback}")
            app.config["SQLALCHEMY_DATABASE_URI"] = fallback
            # dispose existing engine so a new one is created using the changed
            # configuration; ``db.init_app`` should not be called again.
            try:
                db.engine.dispose()
            except Exception:
                pass
            try:
                db.create_all()
            except Exception as exc2:
                # give up after second failure
                app.logger.error(f"fallback create_all also failed: {exc2}")
        # ensure additional columns introduced after initial schema creation
        # (e.g. roll_number, semester, photo on Student) exist.
        try:
            from sqlalchemy import inspect, text
            insp = inspect(db.engine)
            if 'student' in insp.get_table_names():
                cols = {c['name'] for c in insp.get_columns('student')}
                with db.engine.connect() as conn:
                    if 'roll_number' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN roll_number VARCHAR(50) UNIQUE'))
                    if 'semester' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN semester VARCHAR(20)'))
                    if 'photo' not in cols:
                        conn.execute(text('ALTER TABLE student ADD COLUMN photo TEXT'))
        except Exception as exc:
            app.logger.warning(f"could not ensure student columns: {exc}")


# allow command-line invocation for convenience
# running this module directly will create any missing tables and then
# start the development server so you can browse the app at
# http://localhost:5000 (or 0.0.0.0 inside the container).
def _kill_other_instances():
    """Terminate any previously running copies of this script.

    This is useful in a development container where `python app.py` may be
    invoked repeatedly and older processes linger, causing "address already
    in use" errors.  We use ``psutil`` to identify other ``app.py`` commands
    and kill them.  If the library is unavailable we simply return silently.
    """
    try:
        import psutil
        import os
        me = os.getpid()
        for proc in psutil.process_iter(['pid', 'cmdline']):
            info = proc.info
            if info['pid'] == me:
                continue
            cmd = info.get('cmdline') or []
            # match on filename to avoid killing unrelated processes
            if any('app.py' in part for part in cmd):
                proc.kill()
    except ImportError:
        # psutil not installed; nothing we can do
        pass
 

if __name__ == "__main__":
    # clean up any earlier runs so the port will be free
    _kill_other_instances()

    application = create_app()
    create_tables(application)
    print("Tables created (if they didn't exist already). Starting server...")
    # `debug=True` is fine for development; turn off in production.
    port = int(os.getenv("PORT", 5000))
    # disable the built-in reloader; it spawns a second process which
    # our _kill_other_instances helper may immediately kill, leading to the
    # mysterious exit-137 behaviour observed in the container.  The dev
    # server will still restart when the file changes if `debug=True` but
    # without the extra process.
    application.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
