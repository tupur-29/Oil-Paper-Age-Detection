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
import tensorflow.lite as tflite

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Transformer Diagnostics", page_icon="⚡", layout="wide")

# --- CUSTOM CSS FOR UI ---
st.markdown("""
    <style>
    .main-title {text-align: center; color: #4A90E2; padding-bottom: 10px; font-weight: 800;}
    .sub-title {text-align: center; color: #888888; margin-bottom: 40px;}
    .center-card {margin: 0 auto; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. DATABASE SETUP
# ==========================================
def init_db():
    conn = sqlite3.connect('transformer_app.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (email TEXT PRIMARY KEY, password TEXT, name TEXT, emp_id TEXT, phone TEXT, role TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS login_history
                 (email TEXT, login_time TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS inspections
                 (email TEXT, timestamp TEXT, prediction TEXT, confidence REAL)''')
    
    # Create Default Admin just in case
    c.execute("SELECT * FROM users WHERE email='admin'")
    if not c.fetchone():
        c.execute("INSERT INTO users VALUES ('admin', ?, 'Master Admin', 'ADMIN-00', 'N/A', 'admin')", 
                  (hash_password('admin123'),))
    conn.commit()
    conn.close()

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

init_db()

# ==========================================
# 2. SESSION STATE ROUTING
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_email = ""
    st.session_state.role = ""
    st.session_state.current_page = "Landing"

def navigate(page):
    st.session_state.current_page = page

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
# 4. PORTAL PAGES (LANDING, LOGIN, SIGNUP)
# ==========================================
def landing_page():
    st.markdown("<h1 class='main-title'>⚡ AI Transformer Health Monitor</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='sub-title'>Centralized Dashboard for Oil-Paper Insulation Diagnostics</h4>", unsafe_allow_html=True)
    
    # Create two distinct distinct cards side by side
    col1, col_gap, col2 = st.columns([1, 0.2, 1])
    
    with col1:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>👨‍🔧 Field Engineer Portal</h3>", unsafe_allow_html=True)
            st.write("For lab technicians and engineers to run image diagnostics and view their reports.")
            st.write("")
            if st.button("Engineer Login", type="primary", use_container_width=True, key="user_login"):
                navigate("User_Login")
                st.rerun()
            if st.button("Engineer Sign Up", use_container_width=True, key="user_signup"):
                navigate("User_Signup")
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>🛠️ System Admin Portal</h3>", unsafe_allow_html=True)
            st.write("For system administrators to manage users and view global plant inspections.")
            st.write("")
            if st.button("Admin Login", type="primary", use_container_width=True, key="admin_login"):
                navigate("Admin_Login")
                st.rerun()
            if st.button("Admin Sign Up", use_container_width=True, key="admin_signup"):
                navigate("Admin_Signup")
                st.rerun()

def signup_page(role="user"):
    role_title = "Field Engineer" if role == "user" else "System Admin"
    st.markdown(f"<h2 class='main-title'>📝 {role_title} Registration</h2>", unsafe_allow_html=True)
    
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type='password')
            name = st.text_input("Full Name")
            emp_id = st.text_input("Employee / Badge ID")
            phone = st.text_input("Contact Number")
            
            st.write("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Register Account", type="primary", use_container_width=True):
                    if email and password and name:
                        conn = sqlite3.connect('transformer_app.db')
                        c = conn.cursor()
                        c.execute("SELECT * FROM users WHERE email=?", (email,))
                        if c.fetchone():
                            st.error("Email already exists in the system!")
                        else:
                            c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)", 
                                      (email, hash_password(password), name, emp_id, phone, role))
                            conn.commit()
                            st.success(f"Registered as {role_title}! You can now login.")
                        conn.close()
                    else:
                        st.warning("Please fill required fields (Email, Password, Name).")
            with c2:
                if st.button("⬅️ Back to Home", use_container_width=True):
                    navigate("Landing")
                    st.rerun()

def login_page(role="user"):
    role_title = "Field Engineer" if role == "user" else "System Admin"
    st.markdown(f"<h2 class='main-title'>🔐 {role_title} Login</h2>", unsafe_allow_html=True)
    
    # Center the form
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        with st.container(border=True):
            email = st.text_input("Email / Username")
            password = st.text_input("Password", type='password')
            
            st.write("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Secure Login", type="primary", use_container_width=True):
                    conn = sqlite3.connect('transformer_app.db')
                    c = conn.cursor()
                    c.execute("SELECT role FROM users WHERE email=? AND password=?", (email, hash_password(password)))
                    result = c.fetchone()
                    if result:
                        db_role = result[0]
                        if db_role != role:
                            st.error(f"Access Denied. This email is registered as a {db_role}, not a {role}.")
                        else:
                            st.session_state.logged_in = True
                            st.session_state.user_email = email
                            st.session_state.role = db_role
                            st.session_state.current_page = "Dashboard"
                            
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            c.execute("INSERT INTO login_history VALUES (?, ?)", (email, now))
                            conn.commit()
                            st.rerun()
                    else:
                        st.error("Invalid credentials. Please try again.")
                    conn.close()
            with c2:
                if st.button("⬅️ Back to Home", use_container_width=True):
                    navigate("Landing")
                    st.rerun()

# ==========================================
# 5. USER DASHBOARD (ENGINEER)
# ==========================================
def user_dashboard():
    conn = sqlite3.connect('transformer_app.db')
    c = conn.cursor()
    c.execute("SELECT name, emp_id FROM users WHERE email=?", (st.session_state.user_email,))
    user_data = c.fetchone()
    conn.close()
    
    st.sidebar.title(f"👨‍🔧 {user_data[0]}")
    st.sidebar.caption(f"**ID:** {user_data[1]} | **Role:** Field Engineer")
    st.sidebar.write("---")
    
    menu = st.sidebar.radio("Navigation Menu", ["🔬 Run AI Analysis", "📄 My Inspections", "🕒 Login History"])
    
    st.sidebar.write("---")
    if st.sidebar.button("Logout", type="primary", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_page = "Landing"
        st.rerun()

    if menu == "🔬 Run AI Analysis":
        run_analysis()
    elif menu == "📄 My Inspections":
        show_user_reports()
    elif menu == "🕒 Login History":
        show_login_history()

def run_analysis():
    st.markdown("## 🔬 Automated Insulation Diagnosis")
    st.write("Upload a microscopic image of the oil-paper sample to run the AI analysis.")
    st.write("---")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        with st.container(border=True):
            uploaded_file = st.file_uploader("Select Image File", type=["jpg", "png", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Sample Preview", use_column_width=True)
                analyze_btn = st.button("🔍 Analyze Sample", type="primary", use_container_width=True)
            
    with col2:
        if uploaded_file is not None and analyze_btn:
            if interpreter is None:
                st.error("AI Model failed to load. Please check system files.")
                return
                
            with st.spinner("Processing through Neural Network..."):
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
                
                with st.container(border=True):
                    st.write("### 📊 Diagnostic Results")
                    if result == "Highly Aged":
                        st.error(f"**Condition:** {result}")
                        st.error(f"**Confidence:** {confidence:.2f}%")
                        st.markdown("⚠️ **CRITICAL:** Insulation severely degraded. Maintenance required.")
                    elif result == "Lightly Aged":
                        st.warning(f"**Condition:** {result}")
                        st.warning(f"**Confidence:** {confidence:.2f}%")
                        st.markdown("⚠️ **WARNING:** Early aging detected. Schedule monitoring.")
                    else:
                        st.success(f"**Condition:** {result}")
                        st.success(f"**Confidence:** {confidence:.2f}%")
                        st.markdown("✅ **HEALTHY:** Insulation is in optimal condition.")
                        
                with st.container(border=True):
                    st.write("**Probability Distribution:**")
                    prob_dict = {name: float(prob) for name, prob in zip(class_names, output_data)}
                    st.bar_chart(prob_dict)
                
                # Save to DB
                conn = sqlite3.connect('transformer_app.db')
                c = conn.cursor()
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO inspections VALUES (?, ?, ?, ?)", 
                          (st.session_state.user_email, now, result, confidence))
                conn.commit()
                conn.close()

# ==========================================
# 6. ADMIN DASHBOARD
# ==========================================
def admin_dashboard():
    st.sidebar.title("🛠️ Admin Control Center")
    st.sidebar.caption(f"Logged in as: {st.session_state.user_email}")
    st.sidebar.write("---")
    
    menu = st.sidebar.radio("Admin Menu", ["System Overview", "Global Inspections", "Manage Users"])
    
    st.sidebar.write("---")
    if st.sidebar.button("Logout", type="primary", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.current_page = "Landing"
        st.rerun()

    conn = sqlite3.connect('transformer_app.db')
    
    if menu == "System Overview":
        st.markdown("## 📈 System Overview")
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            user_count = pd.read_sql_query("SELECT COUNT(*) FROM users", conn).iloc[0,0]
            insp_count = pd.read_sql_query("SELECT COUNT(*) FROM inspections", conn).iloc[0,0]
            crit_count = pd.read_sql_query("SELECT COUNT(*) FROM inspections WHERE prediction='Highly Aged'", conn).iloc[0,0]
            
            c1.metric("Total Registered Users", user_count)
            c2.metric("Total Inspections Logged", insp_count)
            c3.metric("Critical Alerts", crit_count)
        
    elif menu == "Global Inspections":
        st.markdown("## 🌍 Network-Wide Inspections")
        df = pd.read_sql_query("SELECT email as User, timestamp as Date_Time, prediction as Result, confidence as Confidence_Percent FROM inspections ORDER BY timestamp DESC", conn)
        st.dataframe(df, use_container_width=True)
        
    elif menu == "Manage Users":
        st.markdown("## 👥 Registered Personnel")
        df = pd.read_sql_query("SELECT name as Name, emp_id as Badge_ID, email as Email, role as Role, phone as Contact FROM users", conn)
        st.dataframe(df, use_container_width=True)
        
    conn.close()

# ==========================================
# SHARED VIEW FUNCTIONS
# ==========================================
def show_user_reports():
    st.markdown("## 📄 My Inspection History")
    conn = sqlite3.connect('transformer_app.db')
    df = pd.read_sql_query("SELECT timestamp as Date_Time, prediction as Result, confidence as Confidence_Percent FROM inspections WHERE email=? ORDER BY timestamp DESC", conn, params=(st.session_state.user_email,))
    conn.close()
    
    if df.empty:
        st.info("No inspection records found in your profile.")
    else:
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report (CSV)", data=csv, file_name="my_inspections.csv", mime="text/csv")

def show_login_history():
    st.markdown("## 🕒 Account Access Log")
    conn = sqlite3.connect('transformer_app.db')
    df = pd.read_sql_query("SELECT login_time as Login_Timestamp FROM login_history WHERE email=? ORDER BY login_time DESC", conn, params=(st.session_state.user_email,))
    conn.close()
    st.dataframe(df, use_container_width=True)

# ==========================================
# MAIN ROUTING ENGINE
# ==========================================
if not st.session_state.logged_in:
    if st.session_state.current_page == "Landing":
        landing_page()
    elif st.session_state.current_page == "User_Login":
        login_page(role="user")
    elif st.session_state.current_page == "User_Signup":
        signup_page(role="user")
    elif st.session_state.current_page == "Admin_Login":
        login_page(role="admin")
    elif st.session_state.current_page == "Admin_Signup":
        signup_page(role="admin")
else:
    if st.session_state.role == "admin":
        admin_dashboard()
    else:
        user_dashboard()
