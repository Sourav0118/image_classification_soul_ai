import streamlit as st
import requests
from PIL import Image

# FastAPI backend URL
FASTAPI_URL = "http://localhost:8000/predict"

# Basic authentication credentials
USERNAME = "admin"
PASSWORD = "password"

# Streamlit app
def main():
    st.title("Image Classification App with FastAPI Backend")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Send the image to the FastAPI backend for prediction
        image_data = uploaded_file.getvalue()

        # Send a POST request to the FastAPI backend
        response = requests.post(
            FASTAPI_URL,
            files={"image": image_data},
            auth=(USERNAME, PASSWORD)
        )

        # Debugging: Print out the raw response for inspection
        # st.write("Response Status Code:", response.status_code)
        # st.write("Response Content:", response.text)

        if response.status_code == 200:
            try:
                # Attempt to parse the JSON response
                prediction = response.json()

                # The prediction value is a string, so display it directly
                st.write(f"Predicted Class: {prediction['predicted_class']}")
            except Exception as e:
                st.error(f"Error parsing response: {e}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")

# Run the app
if __name__ == "__main__":
    main()
