import streamlit as st
import numpy as np
from PIL import Image

# Use tflite-runtime instead of full TensorFlow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
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
        interpreter = tflite.Interpreter(model_path="model.tflite")
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
""")
