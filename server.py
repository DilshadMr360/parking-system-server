from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import cv2
import time
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
from ultralytics import YOLO
import cloudinary
from cloudinary.uploader import upload

app = Flask(__name__)
CORS(app)

# Configure Cloudinary
cloudinary.config( 
    cloud_name="dsnznxazm", 
    api_key="525655828417452", 
    api_secret="xgXOQa6FCKFaoS4qTKJ0GYioR8U" 
)

def count_bounding_boxes(detections):
    return sum(len(pred) for pred in detections)


def track_bounding_boxes(detections):
    return count_bounding_boxes(detections)


@app.route('/')
def index():
    return 'Welcome to the Number Plate Detection API'


@app.route('/webcam', methods=['POST'])
def process_webcam():
    # Capture video from webcam
    cap = cv2.VideoCapture(0)

    # Set the processing duration to 30 seconds
    processing_duration = 5  # in seconds

    # Get original video properties
    # frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_rate = 12
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a list to store processed frames
    processed_frames = []

    # Create a folder to save processed frames
    frame_folder = os.path.join(os.getcwd(), "Webcam_smart_parking_system", "frames")
    os.makedirs(frame_folder, exist_ok=True)

    # Define font and text position for annotations
    font = ImageFont.truetype("arial.ttf", 20)
    text_position = (10, 20)

    # Process video frames and perform detection in real-time
    frame_count = 0
    frames_to_process = frame_rate * processing_duration
    while True:
        
        if frame_count >= frames_to_process:
            break 
        print(frame_count)
        ret, frame = cap.read()
    
        if not ret:
            break

        # Perform detection on the frame
        image = Image.fromarray(frame)
        detections_numberplate = yolo_numberplate.predict(image, save=False)

        # Print the number of bounding boxes detected for each object
        num_boxes_numberplate = count_bounding_boxes(detections_numberplate)
        print("Frame {}: Number plate bounding boxes: {}".format(frame_count, num_boxes_numberplate))

        # Track bounding boxes
        num_tracked_boxes_numberplate = track_bounding_boxes(detections_numberplate)
        print("Frame {}: Tracked number plate boxes: {}".format(frame_count, num_tracked_boxes_numberplate))

        # Plot bounding boxes on the frame
        res_plotted = detections_numberplate[0].plot()  # Assuming there's only one detection
        frame_with_boxes = np.array(res_plotted)

        # Add text annotations to the frame
        image_with_text = Image.fromarray(frame_with_boxes)
        draw = ImageDraw.Draw(image_with_text)
        draw.text(text_position, f"Bboxes: {num_boxes_numberplate}, Tracked NumberPlate: {num_tracked_boxes_numberplate}", font=font, fill=(255, 255, 255))

        # Append processed frame to list
        processed_frames.append(np.array(image_with_text))

        # Save processed frame to the frames folder
        frame_path = os.path.join(frame_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(np.array(image_with_text), cv2.COLOR_RGB2BGR))
        frame_count += 1

    # Release the VideoCapture object
    cap.release()

    # Save the processed frames as a video file
    output_video_path = os.path.join(frame_folder, "processed_video.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
    for frame in processed_frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    # Upload video to Cloudinary
    cloudinary_response = upload(output_video_path, resource_type="video")

    # Get URL & ID for the processed video from Cloudinary response
    processed_video_url = cloudinary_response['secure_url']
    processed_public_id = cloudinary_response['public_id']

    return jsonify({
        'processed_video_url': processed_video_url,
        'processed_public_id': processed_public_id,
        'message': 'Processed video and frames saved successfully'
    }), 200


def process_video(video_source, processing_duration=None, start_time=None, input_width=None, input_height=None, frame_rate=None):
    if isinstance(video_source, str):
        # If video source is a file path
        cap = cv2.VideoCapture(video_source)
        video_type = 'file'
    else:
        # If video source is a VideoCapture object (webcam)
        cap = video_source
        video_type = 'webcam'

    if input_width is None or input_height is None or frame_rate is None:
        # Get original video properties
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

    # Define output video dimensions
    output_width = input_width
    output_height = input_height

    # Create a list to store processed frames
    processed_frames = []

    # Create a folder to save processed frames
    frame_folder = os.path.join(os.getcwd(), "SmartParkingSystem", "frames")
    os.makedirs(frame_folder, exist_ok=True)

    # Define font and text position for annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position = (10, 20)
    font_scale = 0.6
    font_color = (255, 255, 255)
    line_type = 2

    # Load YOLO models for number plate  and parkng slot detection
    yolo_number_plate = YOLO('./number_plate_1.pt')  # for detecting number plate
    yolo_parking_slot = YOLO('./parkingslot.pt')  # for detecting  parking slot

    # Check if CUDA is available
    if torch.cuda.is_available():
        yolo_number_plate.model.cuda()
        yolo_parking_slot.model.cuda()

    # Process video frames and perform detection
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection on the frame
        image = np.array(frame)

        # Detect number plate
        detections_number_plate = yolo_number_plate.predict(image, save=False)

        # Detect parking slot
        detections_parkig_slots = yolo_parking_slot.predict(image, save=False)

        # Combine detections
        detections = [detections_number_plate[0], detections_parkig_slots[0]]

        number_plate_boxes = count_bounding_boxes(detections_number_plate)
        print("Number of Number Plate bounding boxes in frame {}: {}".format(frame_count, number_plate_boxes))

        # Track bounding boxes
        num_tracked_boxes = track_bounding_boxes(detections)
        print("Number of tracked bounding boxes in frame {}: {}".format(frame_count, num_tracked_boxes))

        # Plot bounding boxes on the frame for number plate
        res_plotted_number_plate = detections[0].plot()
        frame_with_boxes_number_plate = np.array(res_plotted_number_plate)

        # Plot bounding boxes on the frame for parking slot
        res_plotted_parking_slot = detections[1].plot()
        frame_with_boxes_parking_slot = np.array(res_plotted_parking_slot)

        # Combine frames with bounding boxes for number plate and parking slot
        # frame_with_boxes = frame_with_boxes_number_plate + frame_with_boxes_parking_slot
        frame_with_boxes = frame_with_boxes_number_plate 

        # Add text annotations to the frame
        image_with_text = Image.fromarray(frame_with_boxes)
        draw = ImageDraw.Draw(image_with_text)
        text = f"Number_Plate: {number_plate_boxes}, Number Plate: {num_tracked_boxes}"
        font = ImageFont.truetype("arial.ttf", 30)
        draw.text(text_position, text, font=font, fill=(255, 255, 255))

        # Convert back to numpy array
        frame_with_boxes = np.array(image_with_text)

        # Resize the frame
        frame_with_boxes_resized = cv2.resize(frame_with_boxes, (output_width, output_height))

        # Append processed frame to list
        processed_frames.append(frame_with_boxes_resized)

        # Save processed frame to the frames folder
        frame_path = os.path.join(frame_folder, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame_with_boxes_resized)

        frame_count += 1

        # Check if processing duration exceeded
        if processing_duration and time.time() - start_time > processing_duration:
            break

    # Release the VideoCapture object if it's a file
    if video_type == 'file':
        cap.release()

    # Save the processed frames as a video file
    output_video_path = os.path.join(frame_folder, "processed_video.mp4")
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (output_width, output_height))
    for frame in processed_frames:
        out.write(frame)
    out.release()

    # Upload video to Cloudinary
    cloudinary_response = upload(output_video_path, resource_type="video")

    # Get URL for the processed video from Cloudinary response
    processed_video_url = cloudinary_response['secure_url']

    return processed_video_url, 'Video processed successfully'


@app.route('/upload_plate', methods=['POST'])
def upload_plate():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.mp4'):
        # Save uploaded video
        video_path = "temp.mp4"
        file.save(video_path)

        # Load YOLO models for number plate detection
        yolo_numberplate = YOLO('./number_plate_1.pt')

        # Check if CUDA is available
        if torch.cuda.is_available():
            yolo_numberplate.model.cuda()

        # Create VideoCapture object
        cap = cv2.VideoCapture(video_path)

        # Get original video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a list to store processed frames
        processed_frames = []

        # Create a folder to save processed frames
        frame_folder = os.path.join(os.getcwd(), "client", "public", "frames")
        os.makedirs(frame_folder, exist_ok=True)

        # Define font and text position for annotations
        font = ImageFont.truetype("arial.ttf", 20)
        text_position = (10, 20)

        # Process video frames and perform detection in real-time
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection on the frame
            image = Image.fromarray(frame)
            detections_numberplate = yolo_numberplate.predict(image, save=False)

            # Print the number of bounding boxes detected for each object
            num_boxes_numberplate = count_bounding_boxes(detections_numberplate)
            print("Frame {}: Number plate bounding boxes: {}".format(frame_count, num_boxes_numberplate))

            # Track bounding boxes
            num_tracked_boxes_numberplate = track_bounding_boxes(detections_numberplate)
            print("Frame {}: Tracked number plate boxes: {}".format(frame_count, num_tracked_boxes_numberplate))

            # Plot bounding boxes on the frame
            res_plotted = detections_numberplate[0].plot()  # Assuming there's only one detection
            frame_with_boxes = np.array(res_plotted)

            # Add text annotations to the frame
            image_with_text = Image.fromarray(frame_with_boxes)
            draw = ImageDraw.Draw(image_with_text)
            draw.text(text_position, f"Bboxes: {num_boxes_numberplate}, Tracked NumberPlate: {num_tracked_boxes_numberplate}", font=font, fill=(255, 255, 255))

            # Append processed frame to list
            processed_frames.append(np.array(image_with_text))

            # Save processed frame to the frames folder
            frame_path = os.path.join(frame_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(np.array(image_with_text), cv2.COLOR_RGB2BGR))
            frame_count += 1

        # Release the VideoCapture object
        cap.release()

        # Save the processed frames as a video file
        output_video_path = os.path.join(frame_folder, "processed_video.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
        for frame in processed_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        # Upload video to Cloudinary
        cloudinary_response = upload(output_video_path, resource_type="video")

        # Get URL & ID for the processed video from Cloudinary response
        processed_video_url = cloudinary_response['secure_url']
        processed_public_id = cloudinary_response['public_id']

        return jsonify({
            'processed_video_url': processed_video_url,
            'processed_public_id': processed_public_id,
            'message': 'Processed video and frames saved successfully'
        }), 200
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .mp4 file'}), 400


@app.route('/upload_parking', methods=['POST'])
def upload_parking():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.mp4'):
        # Save uploaded video
        video_path = "temp.mp4"
        file.save(video_path)

        # Load YOLO model for parking slot detection
        yolo_parkingslot = YOLO('./parkingslot.pt')

        # Check if CUDA is available
        if torch.cuda.is_available():
            yolo_parkingslot.model.cuda()

        # Create VideoCapture object
        cap = cv2.VideoCapture(video_path)

        # Get original video properties
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a list to store processed frames
        processed_frames = []

        # Create a folder to save processed frames
        frame_folder = os.path.join(os.getcwd(), "client", "public", "frames")
        os.makedirs(frame_folder, exist_ok=True)

        # Define font and text position for annotations
        font = ImageFont.truetype("arial.ttf", 20)
        text_position = (10, 20)

        # Process video frames and perform detection in real-time
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection on the frame
            image = Image.fromarray(frame)
            detections_parkingslot = yolo_parkingslot.predict(image, save=False)

            # Print the number of bounding boxes detected for each object
            num_boxes_parkingslot = count_bounding_boxes(detections_parkingslot)
            print("Frame {}: Parking slot bounding boxes: {}".format(frame_count, num_boxes_parkingslot))

            # Track bounding boxes
            num_tracked_boxes_parkingslot = track_bounding_boxes(detections_parkingslot)
            print("Frame {}: Tracked parking slot boxes: {}".format(frame_count, num_tracked_boxes_parkingslot))

            # Plot bounding boxes on the frame
            res_plotted = detections_parkingslot[0].plot()  # Assuming there's only one detection
            frame_with_boxes = np.array(res_plotted)

            # Add text annotations to the frame
            image_with_text = Image.fromarray(frame_with_boxes)
            draw = ImageDraw.Draw(image_with_text)
            draw.text(text_position, f"Bboxes: {num_boxes_parkingslot}, Tracked Parking Slots: {num_tracked_boxes_parkingslot}", font=font, fill=(255, 255, 255))

            # Append processed frame to list
            processed_frames.append(np.array(image_with_text))

            # Save processed frame to the frames folder
            frame_path = os.path.join(frame_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_path, cv2.cvtColor(np.array(image_with_text), cv2.COLOR_RGB2BGR))
            frame_count += 1

        # Release the VideoCapture object
        cap.release()

        # Save the processed frames as a video file
        output_video_path = os.path.join(frame_folder, "processed_video.mp4")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))
        for frame in processed_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        # Upload video to Cloudinary
        cloudinary_response = upload(output_video_path, resource_type="video")

        # Get URL & ID for the processed video from Cloudinary response
        processed_video_url = cloudinary_response['secure_url']
        processed_public_id = cloudinary_response['public_id']

        return jsonify({
            'processed_video_url': processed_video_url,
            'processed_public_id': processed_public_id,
            'message': 'Processed video and frames saved successfully'
        }), 200
    else:
        return jsonify({'error': 'Invalid file format. Please upload a .mp4 file'}), 400


# Load YOLO model for number plate detection
print("Loading number plate detection model...")
yolo_numberplate = YOLO('./number_plate_1.pt')
print("Number plate detection model loaded successfully.")

# Load YOLO model for parking slot detection
print("Loading parking slot detection model...")
yolo_parkingslot = YOLO('./parkingslot.pt')
print("Parking slot detection model loaded successfully.")


if __name__ == '__main__':
    app.run(debug=False)
