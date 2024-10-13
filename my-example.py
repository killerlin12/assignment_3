from jetson.inference import detectNet
from jetson.utils import videoSource, videoOutput
import jetson.inference
import jetson.utils

# Initialize the object detection network
net = detectNet("ssd-mobilenet-v2", threshold=0.8)
# Set up the display output
display = videoOutput("display://0")  # 'my_video.mp4' for file

img_path = "/home/nvidia/assignment3/person.jpeg"
img = jetson.utils.loadImage(img_path)
if img is None:  # capture timeout
    print("nothing to detect!!")
    KeyboardInterrupt

while True:
    # Perform object detection
    detections = net.Detect(img)

    # Render the image with bounding boxes and labels
    display.Render(img)

    for detection in detections:
        print(detection)

        # Update the display with network FPS
        # display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
