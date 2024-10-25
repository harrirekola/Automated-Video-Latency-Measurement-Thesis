from pyzbar import pyzbar
import argparse
import datetime
import cv2
from pypylon import pylon
import pandas as pd
import matplotlib.pyplot as plt

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", type=str, default="barcodes.csv",
                help="path to output CSV file containing barcodes")
args = vars(ap.parse_args())

# Initialize the pylon camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set camera parameters
camera.AcquisitionFrameRateEnable.Value = True
camera.AcquisitionFrameRate.Value = 100.0
camera.ExposureTime.Value = 1500.0
camera.MaxNumBuffer = 10

# Start grabbing images from the camera
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# Set up the image converter to convert Pylon images to OpenCV-compatible format
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

csv = open(args["output"], "w")
found = set()
qr_frames = []

while camera.IsGrabbing():
    # Grab the latest image
    grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grab_result.GrabSucceeded():
        # Convert the image to OpenCV format
        image = converter.Convert(grab_result)
        frame = image.GetArray()

        frame = cv2.resize(frame, (190, int(frame.shape[0] * (190 / frame.shape[1]))))

        barcodes = pyzbar.decode(frame)

        for barcode in barcodes:
            # Extract the bounding box location of the barcode and draw the bounding box surrounding the barcode
            (x, y, w, h) = barcode.rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Decode the barcode data
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type

            # Draw the barcode data and barcode type on the image
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # If the barcode text is currently not in our CSV file, write the timestamp + barcode to disk and update the set
            if barcodeData not in found:
                csv.write("{},{}\n".format(datetime.datetime.now(), barcodeData))
                csv.flush()
                found.add(barcodeData)

            # Store frame number and detection time for QR code data processing
            qr_frames.append({
                'frame_number': barcodeData,  # The barcode data containing the timestamp and frame
                'time': datetime.datetime.now()  # The time when the QR code was detected
            })

        # Show the output frame
        cv2.imshow("Barcode Scanner", frame)

        # Break the loop if 'q' key is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Release the grab result
    grab_result.Release()

csv.close()
cv2.destroyAllWindows()
camera.StopGrabbing()
camera.Close()

# Load data
data = pd.read_csv('barcodes.csv')

# Parse the first column as timestamps
data['Timestamp'] = pd.to_datetime(data.iloc[:, 0])

# Extract frame number and QR Code timestamp from the second column
data['Frame'] = data.iloc[:, 1].str.extract(r'Frame:\s(\d+)').astype(int)
data['QR_Timestamp'] = pd.to_datetime(data.iloc[:, 1].str.extract(r'QR Code:\s([\d\-:\.]+)').iloc[:, 0])

# Discard the first 200 frames
data = data[data['Frame'] > 200]

# Calculate delay for each frame (difference between consecutive timestamps)
data['Delay (ms)'] = data['Timestamp'].diff().dt.total_seconds() * 1000

data = data.dropna()

# Calculate statistics
min_delay = data['Delay (ms)'].min()
max_delay = data['Delay (ms)'].max()
avg_delay = data['Delay (ms)'].mean()
std_delay = data['Delay (ms)'].std()

# Calculate skipped frames and accuracy
def calculate_skipped_frames_and_accuracy(data):
    # Extract frame numbers
    frame_numbers = data['Frame'].tolist()

    # Calculate total frames
    total_frames = max(frame_numbers) - min(frame_numbers) + 1

    # Calculate skipped frames
    skipped_frames = []
    for i in range(len(frame_numbers) - 1):
        current_frame = frame_numbers[i]
        next_frame = frame_numbers[i + 1]

        # Check if there's a gap
        if next_frame != current_frame + 1:
            skipped = list(range(current_frame + 1, next_frame))
            skipped_frames.extend(skipped)

    number_of_skipped_frames = len(skipped_frames)

    # Calculate accuracy percentage
    accuracy_percentage = (1 - number_of_skipped_frames / total_frames) * 100 if total_frames > 0 else 0

    return accuracy_percentage

accuracy = calculate_skipped_frames_and_accuracy(data)

plt.figure(figsize=(10, 6))
plt.plot(data['Frame'], data['Delay (ms)'], color='b')

plt.title(f"Min Delay = {min_delay:.0f}ms, Max Delay = {max_delay:.0f}ms, "
          f"Average Delay = {avg_delay:.0f}ms, Std = {std_delay:.0f}ms, "
          f"Accuracy = {accuracy:.2f}%")

plt.xlabel('Frames')
plt.ylabel('Delay (ms)')

plt.grid(True)

plt.show()
