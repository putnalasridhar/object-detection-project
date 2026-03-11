import cv2
import time
import gradio as gr
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

prev_time = 0

def detect(frame):
    global prev_time

    # Convert RGB to BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    results = model(frame)
    boxes = results[0].boxes

    object_count = {}

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in object_count:
            object_count[label] += 1
        else:
            object_count[label] = 1

    frame = results[0].plot()

    # FPS calculation
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Object count display
    y = 80
    text_output = ""

    for obj, count in object_count.items():
        cv2.putText(
            frame,
            f"{obj}: {count}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        text_output += f"{obj}: {count}\n"
        y += 30

    # Convert back to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame, text_output


interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs=[
        gr.Image(label="Detection Output"),
        gr.Textbox(label="Detected Objects")
    ],
    title="Real-Time Object Detection using YOLOv8",
    description="Web based Object Detection System"
)

interface.launch(share=True)