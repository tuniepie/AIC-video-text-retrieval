from PIL import Image
# Open the image file
image = Image.open("/home/hoangtv/Desktop/Attention/Nguyen/utils/static/images/000529.jpg") 
print(type(image))

# Set the desired DPI (dots per inch)
# new_dpi = 72  # Replace with your desired DPI value

# Calculate the new size in pixels based on the DPI
# new_width = int(image.width * (new_dpi / image.info['dpi'][0]))
# new_height = int(image.height * (new_dpi / image.info['dpi'][1]))

# Resize the image to the new size
# resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

# Set the new DPI for the image
# resized_image.info['dpi'] = (new_dpi, new_dpi)

# Save the resized image
# resized_image.save("output_image.jpg")  # Replace "output_image.jpg" with your desired output file path