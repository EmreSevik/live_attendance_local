from flask import Flask, render_template_string, redirect, url_for, flash, request
import subprocess
import csv
import os
import pickle
import face_recognition

app = Flask(__name__)
app.secret_key = 'yoklama123'

HOME_HTML = '''
<!doctype html>
<html lang="tr">
  <head>
    <title>Attendance System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
      .main-btn { width: 220px; height: 220px; font-size: 1.7rem; margin: 30px; }
      .centered { display: flex; justify-content: center; align-items: center; height: 72vh; }
    </style>
  </head>
  <body>
    <nav class="navbar bg-dark">
      <div class="container-fluid justify-content-center">
        <a class="navbar-brand text-white mx-3" href="/">MainPage</a>
        <a class="navbar-brand text-white mx-3" href="/dashboard">Dashboard</a>
        <a class="navbar-brand text-white mx-3" href="/add_user">Add User</a>
      </div>
    </nav>
    <div class="container mt-4">
      {% with messages = get_flashed_messages() %}
        {% if messages %}
          <div class="alert alert-success alert-dismissible fade show" role="alert">
            {{ messages[0] }}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
          </div>
        {% endif %}
      {% endwith %}
    </div>
    <div class="centered">
      <form method="post" action="/run/entrance">
        <button class="btn btn-success main-btn" type="submit">ENTRANCE<br>ATTENDANCE</button>
      </form>
      <form method="post" action="/run/exit">
        <button class="btn btn-danger main-btn" type="submit">EXIT<br>ATTENDANCE</button>
      </form>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
'''

DASHBOARD_HTML = '''
<!doctype html>
<html lang="tr">
  <head>
    <title>Dashboard - Attendance System</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    <nav class="navbar bg-dark">
      <div class="container-fluid justify-content-center">
        <a class="navbar-brand text-white mx-3" href="/">MainPage</a>
        <a class="navbar-brand text-white mx-3" href="/dashboard">Dashboard</a>
        <a class="navbar-brand text-white mx-3" href="/add_user">Add User</a>
      </div>
    </nav>
    <div class="container mt-4">
      <h3 class="mb-4">Attendance Table</h3>
      <table class="table table-bordered table-striped table-hover">
        <thead>
          <tr>
            {% for h in headers %}
              <th>{{h}}</th>
            {% endfor %}
          </tr>
        </thead>
        <tbody>
        {% for row in data %}
          <tr>
            {% for cell in row %}
              <td>{{cell}}</td>
            {% endfor %}
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </body>
</html>
'''

ADD_USER_HTML = '''
<!doctype html>
<html lang="tr">
  <head>
    <title>Add User</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body>
    <nav class="navbar bg-dark">
      <div class="container-fluid justify-content-center">
        <a class="navbar-brand text-white mx-3" href="/">MainPage</a>
        <a class="navbar-brand text-white mx-3" href="/dashboard">Dashboard</a>
        <a class="navbar-brand text-white mx-3" href="/add_user">Add User</a>
      </div>
    </nav>
    <div class="container mt-4">
      <h3 class="mb-4">Add New User (Face)</h3>
      {% with messages = get_flashed_messages() %}
        {% if messages %}
          <div class="alert alert-info">{{ messages[0] }}</div>
        {% endif %}
      {% endwith %}
      <form method="post" enctype="multipart/form-data">
        <div class="mb-3">
          <label class="form-label">Name/ID</label>
          <input type="text" class="form-control" name="username" required>
        </div>
        <div class="mb-3">
          <label class="form-label">Face Image</label>
          <input type="file" class="form-control" name="face_image" accept="image/*" required>
        </div>
        <button class="btn btn-primary" type="submit">Add User</button>
      </form>
    </div>
  </body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/dashboard')
def dashboard():
    data = []
    headers = []
    if os.path.exists('attendance.csv'):
        with open('attendance.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            headers = next(reader)
            for row in reader:
                data.append(row)
    else:
        headers = ["id", "isim", "giris", "cikis", "sure"]
    return render_template_string(DASHBOARD_HTML, headers=headers, data=data)

@app.route('/add_user', methods=['GET', 'POST'])
def add_user():
    if request.method == 'POST':
        username = request.form['username']
        file = request.files['face_image']
        if not file or file.filename == '':
            flash("Please select a file!")
            return redirect(url_for('add_user'))
        # Dosyayı kaydet
        file_path = f"tmp_{username}.jpg"
        file.save(file_path)
        img = face_recognition.load_image_file(file_path)
        face_locs = face_recognition.face_locations(img)
        if not face_locs:
            os.remove(file_path)
            flash("No face detected. Please try another image.")
            return redirect(url_for('add_user'))
        encoding = face_recognition.face_encodings(img, face_locs)[0]
        # face_db.pickle güncelle
        if os.path.exists("face_db.pickle"):
            with open("face_db.pickle", "rb") as f:
                encodings, names, ids = pickle.load(f)
        else:
            encodings, names, ids = [], [], []
        new_id = f"{len(ids)+1:03d}"
        encodings.append(encoding)
        names.append(username)
        ids.append(new_id)
        with open("face_db.pickle", "wb") as f:
            pickle.dump((encodings, names, ids), f)
        os.remove(file_path)
        flash("New user added successfully!")
        return redirect(url_for('add_user'))
    return render_template_string(ADD_USER_HTML)

@app.route('/run/<what>', methods=['POST'])
def run_script(what):
    if what == 'entrance':
        subprocess.Popen(["python3", "entrance.py"])
        flash("Entrance attendance screen angle. You will successfully record the discovery in the windows.")
    elif what == 'exit':
        subprocess.Popen(["python3", "exit.py"])
        flash("Exit attendance screen angle. You will successfully record the discovery in the windows.")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
