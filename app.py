import os
import numpy as np
import gradio as gr
from PIL import Image

from tensorflow import keras

MODEL_PATH = os.path.join("models", "mnist_cnn.keras")


def build_cnn():
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            keras.layers.Conv2D(32, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Dropout(0.25),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"[INFO] Loading cached model: {MODEL_PATH}")
        return keras.models.load_model(MODEL_PATH)

    print("[INFO] No cached model found. Training a new one...")

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    model = build_cnn()
    model.fit(
        x_train,
        y_train,
        epochs=4,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=1,
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[INFO] Model saved to: {MODEL_PATH}")
    return model

def normalize_gradio_image_input(x):
    """
    Gradio 6.x:
    - gr.Image(type="numpy") -> numpy array
    - gr.Sketchpad() -> dict (通常包含 "composite" 或 "layers")
    这里统一提取出真正的图像数组。
    """
    if x is None:
        return None

    # Sketchpad: dict
    if isinstance(x, dict):
        if "composite" in x and x["composite"] is not None:
            return np.array(x["composite"])

        if "layers" in x and x["layers"]:
            return np.array(x["layers"][-1])

        raise ValueError(f"Unsupported Sketchpad dict keys: {list(x.keys())}")

    return np.array(x)

def to_grayscale_uint8(img: np.ndarray) -> np.ndarray:
    img = np.array(img)
    if img.ndim == 3:
        img = img.mean(axis=2)
    return img.astype("uint8")


def auto_invert_if_needed(gray: np.ndarray) -> np.ndarray:
    if gray.mean() > 127:
        return 255 - gray
    return gray


def find_bbox(fg: np.ndarray):
    ys, xs = np.where(fg)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return xs.min(), ys.min(), xs.max(), ys.max()


def preprocess_level2(img: np.ndarray):
    gray = to_grayscale_uint8(img)
    gray = auto_invert_if_needed(gray)

    fg = gray > 50
    bbox = find_bbox(fg)
    if bbox is None:
        preview = np.zeros((28, 28), dtype="uint8")
        x = (preview.astype("float32") / 255.0)[..., None]
        return x, preview

    x0, y0, x1, y1 = bbox

    margin = 5
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(gray.shape[1] - 1, x1 + margin)
    y1 = min(gray.shape[0] - 1, y1 + margin)

    crop = gray[y0 : y1 + 1, x0 : x1 + 1]

    pil = Image.fromarray(crop)
    w, h = pil.size
    target = 20
    if w >= h:
        new_w = target
        new_h = max(1, int(round(h * target / w)))
    else:
        new_h = target
        new_w = max(1, int(round(w * target / h)))

    pil = pil.resize((new_w, new_h))
    small = np.array(pil).astype("uint8")

    canvas = np.zeros((28, 28), dtype="uint8")
    top = (28 - new_h) // 2
    left = (28 - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = small

    preview = canvas
    x = (preview.astype("float32") / 255.0)[..., None]
    return x, preview


model = get_or_train_model()


def predict(sketchpad_img, upload_img):
    raw = sketchpad_img if sketchpad_img is not None else upload_img
    img = normalize_gradio_image_input(raw)

    if img is None:
        return "请在画板写一个数字，或上传一张图片。", np.zeros((28, 28), dtype="uint8")

    x, preview = preprocess_level2(img)
    prob = model.predict(x[None, ...], verbose=0)[0]
    pred = int(np.argmax(prob))

    top3 = np.argsort(prob)[::-1][:3]
    top3_str = ", ".join([f"{i}: {prob[i]:.3f}" for i in top3])
    text = f"预测结果: {pred}\nTop-3 概率: {top3_str}"
    return text, preview


with gr.Blocks() as demo:
    gr.Markdown("# MNIST 手写数字识别 Demo（CNN + 预处理增强 + 模型缓存）")
    gr.Markdown("支持：画板手写 / 上传图片；并展示预处理后的 28×28 输入预览。")

    with gr.Row():
        sketch = gr.Sketchpad(label="画板手写（推荐）")   # 不要 shape 参数
        upload = gr.Image(label="上传图片", type="numpy")

    btn = gr.Button("开始识别")

    with gr.Row():
        out_text = gr.Textbox(label="识别结果", lines=3)
        out_preview = gr.Image(label="预处理后 28×28 预览", type="numpy")

    btn.click(fn=predict, inputs=[sketch, upload], outputs=[out_text, out_preview])

if __name__ == "__main__":
    demo.launch()
