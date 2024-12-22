import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tempfile
import os
from ultralytics import YOLO
from collections import deque

def plt_show(image, title=""):
    st.image(image, caption=title, use_column_width=True)

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def detect_boxes(image):
    """
    Detect boxes in the image using edge detection and contour detection.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Use Canny Edge Detection
    edges = cv2.Canny(blurred_image, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        # Approximate the contour to simplify it
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Consider only rectangular contours (approximation with 4 points)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            boxes.append((x, y, w, h))

    return boxes

def draw_boxes(image, boxes):
    """
    Draw the detected boxes on the image.
    """
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    plt_show(image, title="Detected Boxes")

def estimate_position_and_orientation(image, boxes):
    """
    Estimate the position (x, y, z) and orientation of each box.
    """
    results = []
    for box in boxes:
        x, y, w, h = box
        # Assuming the z-position is based on the size of the box (depth estimation)
        z = w * h  # Placeholder for actual depth estimation
        orientation = "upright"  # Placeholder for actual orientation calculation
        results.append({"position": (x, y, z), "orientation": orientation})

        # Annotate the image
        cv2.putText(image, f"Pos: {x},{y},{z}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Orientation: {orientation}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt_show(image, title="Position and Orientation Estimation")
    return results

def real_time_box_detection(video_path):
    """
    Process a video in real-time to detect boxes, and estimate position and orientation.
    """
    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('7205541-hd_1080_1920_30fps (1).mp4', fourcc, 20.0, 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect boxes
        boxes = detect_boxes(frame)

        # Draw boxes and estimate position/orientation
        annotated_frame = frame.copy()
        positions_orientations = estimate_position_and_orientation(annotated_frame, boxes)

        out.write(annotated_frame)
        cv2.imshow('Real-Time Box Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    st.set_page_config(page_title="Real-Time Box Detection", page_icon=":package:")
    st.title("Box Detection and Position/Orientation Estimation")

    # Image Processing Section
    st.markdown("## Image Box Detection")
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        image_np = np.array(image)

        st.image(image_np, caption="Uploaded Image", use_column_width=True)

        # Detect boxes
        boxes = detect_boxes(image_np)

        if boxes:
            draw_boxes(image_np.copy(), boxes)
            results = estimate_position_and_orientation(image_np.copy(), boxes)
            st.json(results)
        else:
            st.warning("No boxes detected.")

    # Video Processing Section
    st.markdown("---")
    st.markdown("## Video Box Detection")
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "avi"])

    if uploaded_video is not None:
        temp_video_path = os.path.join(tempfile.gettempdir(), "uploaded_video.mp4")
        with open(temp_video_path, "wb") as temp_video:
            temp_video.write(uploaded_video.read())

        real_time_box_detection(temp_video_path)
        st.success("Video processed. Check the output file.")

if __name__ == "__main__":
    main()
