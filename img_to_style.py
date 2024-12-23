from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import os

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

dir_path = "/home/ado/storage/interior/img"

for filename in os.listdir(dir_path):
    if filename.endswith(".jpg"):
        image = Image.open(os.path.join(dir_path, filename))
        question = "What is the style of the room?"

        inputs = processor(image, question, return_tensors="pt")
        output = model.generate(**inputs)
        answer = processor.decode(output[0], skip_special_tokens=True)

        if not os.path.exists(os.path.join(dir_path, "style")):
            os.mkdir(os.path.join(dir_path, "style"))
        # save answer to file
        with open(os.path.join(dir_path, "style", filename[:-4] + ".txt"), "w") as f:
            f.write(answer)
