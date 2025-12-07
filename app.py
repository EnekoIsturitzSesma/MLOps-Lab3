import gradio as gr
import requests
import io

API_URL = "https://lab3-container-latest.onrender.com/predict"

def obtain_pred(image):
    try:
        # Convert PIL image to bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)  # important: move to start of buffer

        files = {"file": ("image.png", buf, "image/png")}
        response = requests.post(API_URL, files=files, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "prediction" in data:
            return f"Predicted label: {data['prediction']}"
        else:
            return f"Error: {data.get('error', 'No prediction returned')}"
    except requests.exceptions.RequestException as e:
        return f"Error contacting API: {e}"

demo = gr.Interface(
    fn=obtain_pred,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Image Classification Demo",
    description="Upload an image to get a random predicted class using the /predict endpoint."
)

if __name__ == "__main__":
    demo.launch()
