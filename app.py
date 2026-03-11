'''import streamlit as st
import numpy as np
from PIL import Image
import tensorflow.lite as tflite


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Insulation Aging Classifier",
    page_icon="⚡",
    layout="centered"
)

# --- HEADER ---
st.title("⚡ Transformer Oil-Paper Aging Detection")
st.markdown("""
This system uses a **Tri-Stream CNN with CBAM Attention** to classify 
microscopic oil-paper insulation samples into aging categories.
""")
st.write("---")

# --- LOAD TFLITE MODEL ---
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tflite.Interpreter(model_path="model (1).tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

interpreter = load_tflite_model()

if interpreter is not None:
    st.success("✅ Model Loaded Successfully!")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# --- PREPROCESSING FUNCTION ---
def preprocess_image(image):
    # 1. Resize to 224x224 (Matching your training input)
    image = image.resize((224, 224))
    
    # 2. Convert to NumPy array
    img_array = np.array(image)
    
    # 3. Ensure 3 channels (RGB) - handle PNG with alpha
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:  # RGBA
        img_array = img_array[..., :3]
    
    # 4. Normalize (Divide by 255.0 as you did in training)
    img_array = img_array.astype('float32') / 255.0
    
    # 5. Add Batch Dimension: Shape becomes (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- PREDICTION FUNCTION ---
def predict(image):
    # Preprocess
    processed_img = preprocess_image(image)
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data[0]

# --- USER INTERFACE ---
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose a microscopic image of oil-paper insulation",
    type=["jpg", "png", "jpeg", "bmp"]
)

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Sample", use_column_width=True)
    
    # Button to Predict
    if st.button("🔍 Analyze Aging Condition", type="primary"):
        if interpreter is None:
            st.error("❌ Model not loaded. Please refresh the page.")
        else:
            with st.spinner("Processing image with Tri-Stream CNN + CBAM..."):
                try:
                    # Run Prediction
                    prediction_array = predict(image)
                    
                    # Class names - VERIFY THIS ORDER MATCHES YOUR TRAINING!
                    # Check your Colab: print(class_indices) to confirm
                    class_names = ["Fresh", "Highly Aged", "Lightly Aged"]
                    
                    pred_index = np.argmax(prediction_array)
                    confidence = np.max(prediction_array) * 100
                    result = class_names[pred_index]
                    
                    # Display Results in the second column
                    with col2:
                        st.write("### 📊 Results")
                        
                        # Show prediction with color coding
                        if result == "Highly Aged":
                            st.error(f"**Prediction: {result}**")
                            st.error(f"**Confidence: {confidence:.2f}%**")
                            st.markdown("⚠️ **CRITICAL:** Insulation is severely degraded. Immediate maintenance recommended.")
                        elif result == "Lightly Aged":
                            st.warning(f"**Prediction: {result}**")
                            st.warning(f"**Confidence: {confidence:.2f}%**")
                            st.markdown("⚠️ **WARNING:** Early signs of aging detected. Schedule inspection.")
                        else:
                            st.success(f"**Prediction: {result}**")
                            st.success(f"**Confidence: {confidence:.2f}%**")
                            st.markdown("✅ **HEALTHY:** Insulation appears fresh and in good condition.")
                    
                    # Show probability distribution
                    st.write("---")
                    st.subheader("📈 Class Probability Distribution")
                    
                    # Create a nice bar chart
                    prob_dict = {name: float(prob) for name, prob in zip(class_names, prediction_array)}
                    st.bar_chart(prob_dict)
                    
                    # Show raw probabilities
                    st.write("**Detailed Probabilities:**")
                    for name, prob in zip(class_names, prediction_array):
                        st.write(f"- {name}: {prob*100:.2f}%")
                    
                except Exception as e:
                    st.error(f"❌ Prediction Error: {e}")

# --- FOOTER ---
st.write("---")
st.markdown("""
**Project:** Transformer Oil-Paper Aging Classification  
**Institution:** National Institute of Technology, Durgapur  
**Department:** Electrical Engineering
""")'''
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import sqlite3
import hashlib
from datetime import datetime

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Transformer AI Software", page_icon="⚡", layout="wide")

# ==========================================
# 1. DATABASE SETUP (SQLite)
# ==========================================
def init_db():
    conn = sqlite3.connect('transformer_app.db')
    c = conn.cursor()
    # Users Table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, password TEXT, name TEXT, emp_id TEXT, phone TEXT, role TEXT)''')
    # Login History Table
    c.execute('''CREATE TABLE IF NOT EXISTS login_history
                 (email TEXT, login_time TEXT)''')
    # Inspection History Table
    c.execute('''CREATE TABLE IF NOT EXISTS inspections
                 (email TEXT, timestamp TEXT, prediction TEXT, confidence REAL)''')
    
    # Create Default Admin if not exists
    c.execute("SELECT * FROM users WHERE email='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES ('admin', ?, 'System Admin', 'ADMIN-01', 'N/A', 'admin')", 
                  (hash_password('admin123'),))
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_email' not in st.session_state:
    st.session_state.user_email = ""
if 'role' not in st.session_state:
    st.session_state.role = ""
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Landing"

init_db()

# ==========================================
# 3. MACHINE LEARNING LOGIC
# ==========================================
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tflite.Interpreter(model_path="model (1).tflite")
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        return None

interpreter = load_tflite_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[-1] == 4:
        img_array = img_array[..., :3]
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ==========================================
# 4. PAGE FUNCTIONS
# ==========================================

def landing_page():
    st.title("⚡ Transformer Oil-Paper Aging Software")
    st.write("Welcome to the AI-based diagnostic dashboard.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Log In (Existing User)", use_container_width=True):
            st.session_state.current_page = "Login"
            st.rerun()
    with col2:
        if st.button("Sign Up (New User)", use_container_width=True):
            st.session_state.current_page = "Signup"
            st.rerun()
    with col3:
        if st.button("Admin Panel", use_container_width=True):
            st.session_state.current_page = "Admin Login"
            st.rerun()

def signup_page():
    st.title("📝 Create a New Account")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    name = st.text_input("Full Name")
    emp_id = st.text_input("Employee ID")
    phone = st.text_input("Phone Number")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Register"):
            conn = sqlite3.connect('transformer_app.db')
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE email=?", (email,))
            if c.fetchone():
                st.error("Email already exists!")
            else:
                c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, 'user')", 
                          (email, hash_password(password), name, emp_id, phone))
                conn.commit()
                st.success("Account created successfully! Please login.")
            conn.close()
    with col2:
        if st.button("Back to Home"):
            st.session_state.current_page = "Landing"
            st.rerun()

def login_page(is_admin=False):
    st.title("🔐 Admin Login" if is_admin else "🔐 User Login")
    email = st.text_input("Email/Username")
    password = st.text_input("Password", type='password')
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            conn = sqlite3.connect('transformer_app.db')
            c = conn.cursor()
            c.execute("SELECT role FROM users WHERE email=? AND password=?", (email, hash_password(password)))
            result = c.fetchone()
            if result:
                role = result[0]
                if is_admin and role != 'admin':
                    st.error("You are not an admin!")
                else:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email
                    st.session_state.role = role
                    st.session_state.current_page = "Dashboard"
                    
                    # Log the login time
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    c.execute("INSERT INTO login_history VALUES (?, ?)", (email, now))
                    conn.commit()
                    st.rerun()
            else:
                st.error("Invalid credentials!")
            conn.close()
    with col2:
        if st.button("Back"):
            st.session_state.current_page = "Landing"
            st.rerun()

def user_dashboard():
    # Fetch User Details
    conn = sqlite3.connect('transformer_app.db')
    c = conn.cursor()
    c.execute("SELECT name, emp_id, phone FROM users WHERE email=?", (st.session_state.user_email,))
    user_data = c.fetchone()
    
    st.sidebar.title(f"Welcome, {user_data[0]}")
    st.sidebar.write(f"**Emp ID:** {user_data[1]}")
    st.sidebar.write("---")
    
    # Navigation
    menu = st.sidebar.radio("Navigation", ["Run AI Analysis", "My Inspection Reports", "My Login History"])
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_page = "Landing"
        st.rerun()

    if menu == "Run AI Analysis":
        run_analysis()
    elif menu == "My Inspection Reports":
        show_user_reports()
    elif menu == "My Login History":
        show_login_history()

def run_analysis():
    st.header("🔬 Run Insulation Diagnosis")
    uploaded_file = st.file_uploader("Upload Microscopic Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Sample", width=300)
        
        if st.button("Analyze", type="primary"):
            if interpreter is None:
                st.error("Model not loaded.")
                return
                
            with st.spinner("Analyzing with Tri-Stream CNN..."):
                processed_img = preprocess_image(image)
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()
                
                interpreter.set_tensor(input_details[0]['index'], processed_img)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                
                class_names = ["Fresh", "Highly Aged", "Lightly Aged"]
                pred_index = np.argmax(output_data)
                result = class_names[pred_index]
                confidence = float(np.max(output_data) * 100)
                
                # Show results
                st.subheader(f"Prediction: {result}")
                st.write(f"Confidence: {confidence:.2f}%")
                
                # SAVE TO DATABASE
                conn = sqlite3.connect('transformer_app.db')
                c = conn.cursor()
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO inspections VALUES (?, ?, ?, ?)", 
                          (st.session_state.user_email, now, result, confidence))
                conn.commit()
                conn.close()
                st.success("✅ Result saved to database.")

def show_user_reports():
    st.header("📄 My Inspection Reports")
    conn = sqlite3.connect('transformer_app.db')
    df = pd.read_sql_query("SELECT timestamp as Date_Time, prediction as Result, confidence as Confidence_Percent FROM inspections WHERE email=?", conn, params=(st.session_state.user_email,))
    conn.close()
    
    if df.empty:
        st.info("No inspections found.")
    else:
        st.dataframe(df, use_container_width=True)
        # CSV Download feature
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report as CSV", data=csv, file_name="my_inspections.csv", mime="text/csv")

def show_login_history():
    st.header("🕒 Login History")
    conn = sqlite3.connect('transformer_app.db')
    df = pd.read_sql_query("SELECT login_time as Login_Timestamp FROM login_history WHERE email=? ORDER BY login_time DESC", conn, params=(st.session_state.user_email,))
    conn.close()
    st.dataframe(df, use_container_width=True)

def admin_dashboard():
    st.sidebar.title("🛠️ Admin Panel")
    menu = st.sidebar.radio("Navigation", ["Global Inspections", "Manage Users"])
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.current_page = "Landing"
        st.rerun()

    conn = sqlite3.connect('transformer_app.db')
    
    if menu == "Global Inspections":
        st.header("🌍 System-Wide Inspection Logs")
        df = pd.read_sql_query("SELECT email as User, timestamp as Date_Time, prediction as Result, confidence as Confidence FROM inspections ORDER BY timestamp DESC", conn)
        st.dataframe(df, use_container_width=True)
        
    elif menu == "Manage Users":
        st.header("👥 Registered Engineers")
        df = pd.read_sql_query("SELECT name, emp_id, email, phone FROM users WHERE role='user'", conn)
        st.dataframe(df, use_container_width=True)
    conn.close()

# ==========================================
# 5. MAIN APP ROUTING
# ==========================================
if not st.session_state.logged_in:
    if st.session_state.current_page == "Landing":
        landing_page()
    elif st.session_state.current_page == "Signup":
        signup_page()
    elif st.session_state.current_page == "Login":
        login_page(is_admin=False)
    elif st.session_state.current_page == "Admin Login":
        login_page(is_admin=True)
else:
    if st.session_state.role == "admin":
        admin_dashboard()
    else:
        user_dashboard()
