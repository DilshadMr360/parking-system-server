from flask import Flask, request, jsonify
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
    cloud_name = "dsnznxazm", 
    api_key = "525655828417452", 
    api_secret = "xgXOQa6FCKFaoS4qTKJ0GYioR8U" 
)


def count_bounding_boxes(detections):
    return sum(len(pred) for pred in detections)


def track_bounding_boxes(detections):
    # Implement your tracking logic here
    # This function should return the number of tracked bounding boxes
    # For now, let's return the same count as detected bounding boxes
    return count_bounding_boxes(detections)


@app.route('/')
def index():
    return 'Welcome to the Number Plate Detection API'


@app.route('/webcam', methods=['POST'])
def handle_webcam():
    # Implement logic to handle webcam data or return an appropriate response
    return jsonify({'message': 'Webcam data received'})


@app.route('/upload_plate', methods=['POST'])
def upload_plate():
    # print ("hello world")
    # return "hello world"
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.mp4'):
        # Save uploaded video
        video_path = "temp.mp4"
        file.save(video_path)

        # Load YOLO models for number plate and parking slot detection
        yolo_numberplate = YOLO('./number_plate_1.pt')
        yolo_parkingslot = YOLO('./parkingslot.pt')

        # Check if CUDA is available
        if torch.cuda.is_available():
            yolo_numberplate.model.cuda()
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
            detections_numberplate = yolo_numberplate.predict(image, save=False )

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
    # print ("hello world")
    # return "hello world"
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
            yolo_numberplate.model.cuda()
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
            print("Frame {}: Parking slot bounding boxes: {}".format(frame_count,num_boxes_parkingslot))

            # Track bounding boxes
            num_tracked_boxes_parkingslot = track_bounding_boxes(detections_parkingslot)
            # print("Frame {}: Tracked number plate boxes: {}, Tracked parking slot boxes: {}".format(frame_count, num_tracked_boxes_parkingslot))
            # print("Frame {}: Tracked number plate boxes: {}, Tracked parking slot boxes: {}".format(frame_count, num_tracked_boxes_numberplate, num_tracked_boxes_parkingslot))
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
