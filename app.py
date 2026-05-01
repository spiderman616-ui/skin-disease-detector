import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Load model
model = tf.keras.models.load_model('skin_disease_model.h5')

# Class names
class_names = ['Atopic Dermatitis', 'Melanoma', 'Tinea Ringworm Candidiasis']

# Fixed risk levels, guidance and precautions per disease
risk_info = {
    'Melanoma': {
        'level': 'Critical',
        'emoji': '🔴',
        'guidance': 'This may require urgent medical attention. Please consult a healthcare professional as soon as possible.',
        'precautions': [
            'Avoid prolonged sun exposure and always wear sunscreen (SPF 30+).',
            'Do not scratch, pick, or irritate the affected area.',
            'Wear protective clothing when going outdoors.',
            'Monitor the area for any changes in size, shape, or color.',
        ]
    },
    'Atopic Dermatitis': {
        'level': 'Moderate',
        'emoji': '🟡',
        'guidance': 'Consider visiting a dermatologist for proper evaluation and care.',
        'precautions': [
            'Keep skin moisturized regularly with fragrance-free creams.',
            'Avoid harsh soaps, detergents, and synthetic fabrics.',
            'Do not scratch the affected area as it may worsen the condition.',
            'Identify and avoid personal triggers such as dust, pollen, or certain foods.',
        ]
    },
    'Tinea Ringworm Candidiasis': {
        'level': 'Low',
        'emoji': '🟢',
        'guidance': 'Keep the area clean and dry. If symptoms persist or worsen, consult a pharmacist or doctor.',
        'precautions': [
            'Keep the affected area clean and dry at all times.',
            'Avoid sharing towels, clothing, or personal items.',
            'Wear loose, breathable clothing to reduce moisture.',
            'Avoid scratching to prevent spreading the infection.',
        ]
    }
}

def generate_gradcam(img_array, model):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer('Conv_1_bn').output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_class = tf.argmax(predictions[0])
        loss = predictions[:, top_class]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_heatmap(original_img, heatmap):
    img = np.array(original_img.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlaid = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)
    return Image.fromarray(overlaid)

def predict(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    top_index = np.argmax(predictions)
    top_class = class_names[top_index]
    confidence = float(predictions[top_index]) * 100
    risk = risk_info[top_class]

    heatmap = generate_gradcam(img_array, model)
    overlaid_image = overlay_heatmap(image, heatmap)

    precautions_list = "\n".join([f"- {p}" for p in risk['precautions']])

    result = f"""
## 🔬 Detection Result

**Detected Disease:** {top_class}

---

### 📊 Confidence Score
**{confidence:.1f}%** confident in this prediction

---

### ⚠️ Risk Level
{risk['emoji']} **{risk['level']} Risk**

---

### 💡 Guidance
{risk['guidance']}

---

### 🛡️ Precautions
{precautions_list}

---

> ⚕️ *This tool is for guidance only and does not replace professional medical advice. Always consult a qualified healthcare professional for diagnosis and treatment.*
"""
    return overlaid_image, result

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Skin Image"),
    outputs=[
        gr.Image(type="pil", label="🔍 Affected Area Highlighted"),
        gr.Markdown(label="Result")
    ],
    title="🩺 Skin Disease Detector",
    description="Upload a skin image to detect the disease, see the affected area highlighted, and get guidance and precautions.",
).launch(server_name="0.0.0.0", server_port=7860)