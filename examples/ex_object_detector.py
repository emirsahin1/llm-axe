from llm_axe.models import OllamaChat
from llm_axe import ObjectDetectorAgent

# Load a multimodal LLM capable of detecting objects in an image
# llava:7b works great.
llm = OllamaChat(model="llava:7b")
detector = ObjectDetectorAgent(llm, llm)


# It will detect objects according to our criteria
# If we want all objects, we can use the detection_criteria="Detect all objects in the image."
resp = detector.detect(images=["../img.jpg"], detection_criteria="Things related to transportation")
print(resp)

# Output Example:
# {
#     "objects": [
#         {
#         "label": "Traffic light",
#         "location": "Intersection",
#         "description": "Three lights (red, yellow, and green)"
#         },
#         {
#         "label": "Street signs",
#         "location": "At least two street signs are present",
#         "description": "One of which appears to be a road sign indicating directions or distances"
#         },
#         {
#         "label": "Street lights",
#         "location": "Both ends of the intersection",
#         "description": "No information"
#         }
#     ]
# }

# If we only want specific objects, we can pass in objects=["sheep", "chicken", "cat", "dog"]
resp = detector.detect(images=["../img2.jpg"], objects=["sheep", "chicken", "cat", "dog"])
print(resp)

# Output
# {
# "objects": [
# {
# "label": "Sheep",
# "location": "In the field",
# "description": "White with black spots"
# },
# {
# "label": "Dog",
# "location": "Near the barn",
# "description": "Brown with white spots"
# }
# ]
# }