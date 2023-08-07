# track_ros
ROS package for tracking using SAM and XMem


## Interactive prompt for SAM
```bash
roslaunch track_anything_ros publish_mask.launch input_image:=/image/topic/name
```

## Use bounding box prompt from Grounding DINO 
```bash
roslaunch track_anything_ros grounded_publish_mask.launch input_image:=/image/topic/name text_prompt:="person . table . plate ."
```
