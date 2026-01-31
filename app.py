import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Insulation Aging Classifier", page_icon="⚡")

# --- HEADER ---
st.title("⚡ Transformer Oil-Paper Aging Detection")
st.markdown("""
This system uses a **Tri-Stream CNN with CBAM Attention** to classify microscopic insulation samples.
""")

# --- LOAD MODEL FUNCTION ---
@st.cache_resource
def load_model():
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

try:
    interpreter = load_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- PREPROCESSING FUNCTION ---
def preprocess_image(image):
    # 1. Resize to 224x224 (Matching your training input)
    image = image.resize((224, 224))
    
    # 2. Convert to NumPy array
    img_array = np.array(image)
    
    # 3. Ensure 3 channels (RGB)
    if img_array.shape[-1] == 4:  # If PNG has alpha channel
        img_array = img_array[..., :3]
        
    # 4. Normalize (Divide by 255.0 as you did in training)
    img_array = img_array.astype('float32') / 255.0
    
    # 5. Add Batch Dimension: Shape becomes (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# --- USER INTERFACE ---
uploaded_file = st.file_uploader("Upload a Microscopic Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Sample", use_column_width=True)
    
    # Button to Predict
    if st.button("Analyze Aging Condition"):
        with st.spinner("Processing image with Tri-Stream CNN..."):
            try:
                # Preprocess
                processed_img = preprocess_image(image)
                
                # Set Input Tensor
                interpreter.set_tensor(input_details[0]['index'], processed_img)
                
                # Run Inference
                interpreter.invoke()
                
                # Get Output Tensor
                output_data = interpreter.get_tensor(output_details[0]['index'])
                
                # Get Class Index and Confidence
                class_names = ["Fresh", "Highly Aged", "Lightly Aged"] # Ensure this order matches your training!
                pred_index = np.argmax(output_data)
                confidence = np.max(output_data) * 100
                result = class_names[pred_index]
                
                # Display Results
                st.write("---")
                st.subheader(f"Prediction: **{result}**")
                st.write(f"Confidence: **{confidence:.2f}%**")
                
                # Visual Bar Chart for Probabilities
                st.write("Class Probabilities:")
                st.bar_chart(dict(zip(class_names, output_data[0])))
                
                # Custom Message based on result
                if result == "Highly Aged":
                    st.error("⚠️ CRITICAL: Insulation is severely degraded. Maintenance recommended.")
                elif result == "Lightly Aged":
                    st.warning("⚠️ WARNING: Early signs of aging detected.")
                else:
                    st.success("✅ HEALTHY: Insulation appears fresh and good.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
