import json, os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

app = Flask(__name__, static_url_path='/static')
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')

# Persistent user store (JSON file)
users_file_path = os.path.join(os.path.dirname(__file__), 'data', 'users.json')

def _load_users():
    try:
        if os.path.exists(users_file_path):
            with open(users_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception:
        pass
    return {}

def _save_users(users_dict):
    os.makedirs(os.path.dirname(users_file_path), exist_ok=True)
    with open(users_file_path, 'w', encoding='utf-8') as f:
        json.dump(users_dict, f)

USERS = _load_users()

def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get('user_id'):
            flash('Please log in to continue.', 'warning')
            return redirect(url_for('login', next=request.path))
        return view_func(*args, **kwargs)
    return wrapped_view

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        user_hash = USERS.get(username)
        if user_hash and check_password_hash(user_hash, password):
            session['user_id'] = username
            next_url = request.args.get('next') or url_for('welcome')
            return redirect(next_url)
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# data directory
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# Open json file
json_file_path = os.path.join(data_dir, 'exercises.json')
with open(json_file_path, 'r', encoding='utf-8') as file:
    exercises = json.load(file)

# Remove folder names in image filenames
for exercise in exercises:
    images = exercise["images"]
    exercise["images"] = [image.split('/')[-1] for image in images]

# Convert the modified exercise data to a pandas DataFrame
dataframe = pd.DataFrame(exercises)

# Save the DataFrame to a CSV file
csv_file_path = os.path.join(data_dir, 'exercises.csv')
dataframe.to_csv(csv_file_path, index=False, sep=',')
csv_cleaned_file_path = os.path.join(data_dir, 'exercises_cleaned.csv')

# Load the cleaned data from the CSV file
df = pd.read_csv(csv_cleaned_file_path)

# Convert the 'images' field from a string to a list and strip single quotes
df['images'] = df['images'].apply(lambda x: [image.strip(" '") for image in x.strip("[]").split(", ")])

# Build an in-memory lookup by exercise id (string keys)
id_to_exercise = {}
for _, row in df.iterrows():
    record = row.to_dict()
    id_to_exercise[str(record.get('id'))] = record

# Define the priority for user input fields
priority_fields = ['primaryMuscles','level', 'equipment', 'secondaryMuscles', 'force', 'mechanic', 'category']

# Define priority weights
priority_weights = [20, 15, 10, 5, 3, 2, 1]

# Concatenate the relevant columns to create content for recommendations
df['content'] = df[priority_fields].apply(
    lambda row: (
        ' '.join([str(val) * weight for val, weight in zip(row, priority_weights)])
    ),
    axis=1
)

# Create a TF-IDF vectorizer to convert the content into numerical form
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['content'])

# Calculate the cosine similarity between exercises
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/welcome')
def welcome_page():
    return render_template('welcome.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        if not username or not password:
            flash('Email and password are required.', 'warning')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'warning')
            return render_template('register.html')
        if USERS.get(username):
            flash('Account already exists. Please login.', 'info')
            return redirect(url_for('login'))
        USERS[username] = generate_password_hash(password)
        _save_users(USERS)
        flash('Account created. Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/beginner', methods=['GET', 'POST'])
def beginner():
    primary_muscles = ["Chest", "Biceps", "Abdominals", "Quadriceps", "Middle Back", "Glutes", "Hamstrings", "Calves "]
    selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle')
    if request.method == 'POST':
        # Handle form submission and update the selected primary muscle
        selected_primary_muscle = request.form.get('selectedPrimaryMuscle')
        # Store the selected primary muscle in the cookie or local storage
        response = redirect(url_for('recommend_exercises'))
        response.set_cookie('selectedPrimaryMuscle', selected_primary_muscle)
        return response
    return render_template('beginner.html', primary_muscles=primary_muscles, selectedPrimaryMuscle=selected_primary_muscle)

@app.route('/advanced', methods=['GET', 'POST'])
def advanced():
    primary_muscles = ["Neck", "Shoulders", "Chest", "Biceps", "Forearms", "Abdominals", "Quadriceps", "Adductors", "Calves",
                       "Traps", "Triceps", "Lats", "Middle Back", "Lower Back", "Abductors", "Glutes", "Hamstrings", "Calves "]

    selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle')
    if request.method == 'POST':
        # Handle form submission and update the selected primary muscle
        selected_primary_muscle = request.form.get('selectedPrimaryMuscle')
        # Store the selected primary muscle in the cookie or local storage
        response = redirect(url_for('recommend_exercises'))
        response.set_cookie('selectedPrimaryMuscle', selected_primary_muscle)
        return response
    return render_template('advanced.html', primary_muscles=primary_muscles, selectedPrimaryMuscle=selected_primary_muscle)

@app.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend_exercises():
    exercise_data = []
    user_input = {}
    selected_primary_muscle= ""
    if request.method == 'POST':
        user_input = {field: request.form.get(field) for field in priority_fields}

        # Retrieve the selected primary muscle from the cookie
        selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle', "")

        for field in priority_fields:
            if user_input[field] is None:
                user_input[field] = ""  # Set to an empty string or a default value

        # Extract and process the secondary muscles
        secondary_muscles = request.form.getlist('secondaryMuscles[]')
        secondary_muscles_str = ' '.join(secondary_muscles)

        user_content = (
            selected_primary_muscle * 20 + ' ' +
            ''.join(map(str, user_input['level'])) * priority_weights[0] + ' ' +
            ''.join(map(str, user_input['equipment'])) * priority_weights[1] + ' ' +
            secondary_muscles_str * priority_weights[2] + ' ' +
            ''.join(map(str, user_input['force'])) * priority_weights[3] + ' ' +
            ''.join(map(str, user_input['mechanic'])) * priority_weights[4] + ' ' +
            ''.join(map(str, user_input['category'])) * priority_weights[5]
        )

        # Convert user content into TF-IDF vector for recommendation
        user_tfidf_matrix = tfidf_vectorizer.transform([user_content])
        user_cosine_sim = linear_kernel(user_tfidf_matrix, tfidf_matrix)
        sim_scores = user_cosine_sim[0]
        exercise_indices = sim_scores.argsort()[::-1][:5]  # Select top 5 recommendations

        # Convert exercise_indices to a list of exercise IDs
        exercise_ids = [str(df.iloc[index]["id"]) for index in exercise_indices]

        for exercise_id in exercise_ids:
            exercise_doc = id_to_exercise.get(exercise_id)
            if exercise_doc:
                # Ensure instructions is always a string
                if 'instructions' in exercise_doc and isinstance(exercise_doc['instructions'], str):
                    # Replace "\n" with "<br>" to add line breaks in the instructions
                    exercise_doc['instructions'] = exercise_doc['instructions'].replace('.,', '<br>')
                else:
                    # Set default instructions if missing or not a string
                    exercise_doc['instructions'] = 'No instructions available.'
                exercise_data.append(exercise_doc)

        # Render the recommendations template with the results
        return render_template('recommendations.html', recommendations=exercise_data, user_input=user_input, selectedPrimaryMuscle=selected_primary_muscle)
    # Handle the case where there's no POST data (initial page load or form submission)
    return render_template('recommendations.html', recommendations=exercise_data, user_input=user_input, selectedPrimaryMuscle=selected_primary_muscle)


# Calculate the cosine similarity between exercises (item-based collaborative filtering)
cosine_sim_items = cosine_similarity(tfidf_matrix.T, tfidf_matrix.T)

@app.route('/more_recommendations', methods=['GET', 'POST'])
@login_required
def more_recommendations():
    exercise_data = []
    selected_primary_muscle = ""

    if request.method == 'POST':
        selected_primary_muscle = request.cookies.get('selectedPrimaryMuscle', "")

        # Retrieve user input data from the hidden input field
        user_input = json.loads(request.form.get('user_input', '{}'))

        # Extract and process the secondary muscles
        secondary_muscles = request.form.getlist('secondaryMuscles[]')
        secondary_muscles_str = ' '.join(secondary_muscles)

        user_content = (
            selected_primary_muscle * 20 + ' ' +
            ''.join(map(str, user_input.get('level', ''))) * priority_weights[0] + ' ' +
            ''.join(map(str, user_input.get('equipment', ''))) * priority_weights[1] + ' ' +
            secondary_muscles_str * priority_weights[2] + ' ' +
            ''.join(map(str, user_input.get('force', ''))) * priority_weights[3] + ' ' +
            ''.join(map(str, user_input.get('mechanic', ''))) * priority_weights[4] + ' ' +
            ''.join(map(str, user_input.get('category', ''))) * priority_weights[5]
        )

        # Convert user content into TF-IDF vector for recommendation
        user_tfidf_matrix = tfidf_vectorizer.transform([user_content])
        user_cosine_sim = cosine_similarity(user_tfidf_matrix, tfidf_matrix)

        # Calculate the similarity between the user's preferences and exercises (item-based collaborative filtering)
        item_sim_scores = cosine_similarity(user_cosine_sim, tfidf_matrix.T)[0]
        
        # Get the indices of exercises based on item similarity
        exercise_indices = item_sim_scores.argsort()[-5:][::-1]

        # Convert exercise_indices to a list of exercise IDs
        exercise_ids = [str(df.iloc[index]["id"]) for index in exercise_indices]

        for exercise_id in exercise_ids:
            exercise_doc = id_to_exercise.get(exercise_id)
            if exercise_doc:
                # Ensure instructions is always a string
                if 'instructions' in exercise_doc and isinstance(exercise_doc['instructions'], str):
                    # Replace "\n" with "<br>" to add line breaks in the instructions
                    exercise_doc['instructions'] = exercise_doc['instructions'].replace('.,', '<br>')
                else:
                    # Set default instructions if missing or not a string
                    exercise_doc['instructions'] = 'No instructions available.'
                exercise_data.append(exercise_doc)

        # Render the more_recommendations template with the results
        return render_template('more_recommendations.html', recommendations=exercise_data, user_input=user_input,
                               selectedPrimaryMuscle=selected_primary_muscle)

    # Handle the case where there's no POST data (initial page load or form submission)
    return render_template('more_recommendations.html', recommendations=exercise_data, user_input=user_input,
                           selectedPrimaryMuscle=selected_primary_muscle)
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=int(os.environ.get('PORT', 5000)))
