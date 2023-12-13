from PIL import Image
import glob

for i, im_path in enumerate(glob.glob("../training_images/*.png")):
     image = Image.open(im_path)
     width, height = image.size
     new_size = (32, 32)
     resized_image = image.resize(new_size)
     resized_image.save(im_path, optimize=True, quality=100)