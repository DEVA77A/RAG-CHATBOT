import os
import shutil
from PIL import Image

def crop_logo():
    original_source = 'C:/Users/Acer/.gemini/antigravity-ide/brain/50920446-1792-46c8-8845-94610cb67e9c/media__1782407204760.jpg'
    img_path = 'd:/RAG/frontend/public/logo.jpg'
    
    if not os.path.exists(original_source):
        print("Original source logo path does not exist!")
        return
        
    # Re-copy the original file to public/logo.jpg to overwrite previous crop
    shutil.copy(original_source, img_path)
    print("Original logo restored.")
    
    img = Image.open(img_path)
    
    # Crop higher up to exclude the "RAG X" text completely
    # x from 320 to 704 (center is 512, width 384)
    # y from 80 to 464 (center is 272, height 384)
    box = (320, 80, 704, 464)
    cropped = img.crop(box)
    
    # Save cropped image
    cropped.save(img_path, quality=95)
    print("Logo cropped higher and saved successfully!")

if __name__ == '__main__':
    crop_logo()
