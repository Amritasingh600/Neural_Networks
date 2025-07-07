import gradio as gr
import numpy as np
from tensorflow import keras
from PIL import Image

# Load your saved model
model = keras.models.load_model('digit_model.keras')

# Define prediction function
def predict_digit(input_data):
    import traceback
    try:
        print("Input type:", type(input_data))
        print("Input keys:", list(input_data.keys()))
        
        # Extract image from 'composite' key instead of 'image'
        if isinstance(input_data, dict) and "composite" in input_data:
            image = input_data["composite"]
            print("Extracted image from 'composite' key")
        else:
            return "Error: 'composite' key not found in input dict"

        from PIL import Image
        import numpy as np

        print("Image type:", type(image))

        img = Image.fromarray(image).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img)
        img_array = 255 - img_array  # invert colors
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        print(f"Prediction: {digit}")
        return f"Predicted Digit: {digit}"

    except Exception as e:
        print("Error in predict_digit:", e)
        traceback.print_exc()
        return f"Error: {str(e)}"

# Create Gradio interface with Sketchpad!
interface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(canvas_size=(200, 200)),
    outputs="text",
    title="Handwritten Digit Recognizer",
    description="Draw a digit (0-9) and let the neural network predict it!"
)
interface.launch()
