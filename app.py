import streamlit as st
import tensorflow as tf
import numpy as np
import random
import google.generativeai as genai

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_cnn_model.h5')

model = load_my_model()

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_test_reshaped = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

try:
    genai.configure(api_key="AIzaSyA3LkzFzU05vNv-FCdbD5Kf3rAgsQ32NaA")
    llm = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    llm = None

def get_model_prediction(image_index):
    """Gets the CNN model's prediction for a given image index."""
    image = x_test_reshaped[image_index]
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_digit = np.argmax(prediction)
    actual_digit = y_test[image_index]
    return predicted_digit, actual_digit

def get_llm_explanation(predicted_digit, user_question):
    """Generates an explanation from the LLM based on the prediction and user question."""
    if not llm:
        return "LLM not configured. Please check your API key."

    prompt = f"""
    You are an AI assistant helping a user understand a machine learning model's prediction.
    The model predicted the handwritten digit is a '{predicted_digit}'.
    The user has a question: '{user_question}'

    Based on the typical visual features of the digit '{predicted_digit}', provide a simple, helpful answer.
    For example, if the digit is 7, mention the horizontal top line and the diagonal stroke.
    If the user just says "why?", explain the key features of the predicted digit.
    """
    try:
        response = llm.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"An error occurred with the LLM: {e}"

#Streamlit App UI 
st.set_page_config(layout="wide")
st.title("🧠 Interactive MNIST Model Study Buddy")

if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = random.randint(0, len(x_test) - 1)

col1, col2 = st.columns([1, 2])

with col1:
    st.header("The Digit")
    if st.button("Show New Random Digit", use_container_width=True):
        st.session_state.current_image_index = random.randint(0, len(x_test) - 1)

    image_index = st.session_state.current_image_index
    st.image(x_test[image_index], width=250, caption=f"Image from test set (Index: {image_index})")

    predicted_digit, actual_digit = get_model_prediction(image_index)
    st.metric(label="Model's Prediction", value=f"{predicted_digit}")
    st.write(f"**Actual Label:** {actual_digit}")
    if predicted_digit == actual_digit:
        st.success("The model was correct!")
    else:
        st.error("The model was incorrect.")

with col2:
    st.header("Chat with the AI Assistant")
    st.markdown("Ask a question about why the model might have made this prediction.")

    user_question = st.text_input("Your question:", "Why did the model predict this?")

    if user_question:
        with st.spinner("The AI assistant is thinking..."):
            explanation = get_llm_explanation(predicted_digit, user_question)
            st.markdown(explanation)