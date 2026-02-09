from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput

net = detectNet("ssd-mobilenet-v2", threshold=0.5)
camera = videoSource("/home/nvidia/jetson-inference/python/examples/01.jpg") 
display = videoOutput("display://0") 
while display.IsStreaming():
  img = camera.Capture()
  if img is None: # capture timeout
    continue
  detections = net.Detect(img)
  print(detections)
  display.Render(img)
  display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
