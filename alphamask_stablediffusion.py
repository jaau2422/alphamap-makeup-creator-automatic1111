from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils

from tkinter import *
from tkinter import filedialog
import tkinter as tk

import os
import json
import base64
import io
import requests


import base64
from PIL import Image, ImageTk
import colorsys



IMG2IMG_URL = 'http://127.0.0.1:7861/sdapi/v1/img2img'




def generate_request(b64image: str, prompt: str, **kwargs):
    """
    Generate a request object from the given input image and prompt.
    """
    return {
        'prompt': prompt,
        'init_images': [b64image],
        **kwargs
    }


def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    return requests.post(url, data=json.dumps(data))


def _b64encode(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")


def img2b64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    return _b64encode(buffered.getvalue())


def save_encoded_image(b64_image: str, output_path: str):
    """
    Save the given image to the given output path.
    """
    
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))

def product_image(labl, product_index):
    # Retrieve data from API
    response = requests.get("https://makeup-api.herokuapp.com/api/v1/products.json?brand=l%27oreal")

    # Parse JSON response and retrieve hex value and product link for product at the specified index
    data = response.json()[product_index]
    image_link = data['image_link']

    # Download the image and display it on the GUI
    img_data = requests.get(image_link).content
    img = Image.open(io.BytesIO(img_data))
    photo_img = ImageTk.PhotoImage(img)
    labl.config(image=photo_img)
    labl.image = photo_img

def result_image(show):
    img = Image.open("inpaint-person.png")
    # Convert the image to a PhotoImage object
    photo_img = ImageTk.PhotoImage(img)
    # Create a label with the image
    show = show.config(image=photo_img)
    show.image = photo_img


def create_variable(label):
    global variable
    variable = label
    print(f"Variable created with value {variable}!")

def update_label_color(color):
    label.config(text=f'you chose {color}!')
    label.pack(side= tk.TOP)


INPAINTING_FILL_METHODS = ['fill', 'original', 'latent_noise', 'latent_nothing']

#create a function to uplaod an image
global filename
filename = ""
def showimage():
    global filename
    filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="select Image", filetype=(("PNG FILE","*.png"),("all file","blibubb.txt")))
    #img_cv=cv2.imread(filename)
    #return img_cv

def takepicture():
    # Open the default camera
    cap = cv2.VideoCapture(0)
    # Capture a single frame
    ret, frame = cap.read()
    # Save the frame as a PNG file
    cv2.imwrite('captured_image.png', frame)
    # Release the camera
    cap.release()
    # Open the saved PNG file with OpenCV
    img = cv2.imread('captured_image.png')
    return img


#Create the winoowwowow
window = tk.Tk()
window.title("Button Example")
window.geometry("1400x900")

    # Add buttons to the window



import argparse
parser = argparse.ArgumentParser(description='Inpaint instances of people using stable '
                                                 'diffusion.')
    #parser.add_argument('img_path', type=str, help='Path to input image.')
parser.add_argument('-o', '--output_path', type=str, default='inpaint-person.png',
                        help='Path to output image.')
parser.add_argument('-p', '--prompt', type=str, default='',
                        help='Stable diffusion prompt to use.')
parser.add_argument('-n', '--negative_prompt', type=str, default='person',
                        help='Stable diffusion negative prompt.')
parser.add_argument('-W', '--width', type=int, default=680, help='Width of output image.')
parser.add_argument('-H', '--height', type=int, default=832, help='Height of output image.')
parser.add_argument('-s', '--steps', type=int, default=30, help='Number of diffusion steps.')
parser.add_argument('-c', '--cfg_scale', type=int, default=25, help='Classifier free guidance '
                        'scale, i.e. how strongly the image should conform to prompt.')
parser.add_argument('-S', '--sample_name', type=str, default='Euler a', help='Name of sampler '
                        'to use.')
parser.add_argument('-d', '--denoising_strength', type=float, default=0.75, help='How much to '
                        'disregard original image.')
parser.add_argument('-f', '--fill', type=str, default=INPAINTING_FILL_METHODS[0],
                        help='The fill method to use for inpainting.')
parser.add_argument('-b', '--mask_blur', type=int, default=6, help='Blur radius of Gaussian '
                        'filter to apply to mask.')
parser.add_argument('-B', '--bounding_box', action='store_true', help='Convert mask to '
                        'bounding box.')
parser.add_argument('-D', '--bbox_dilation', type=float, default=16, help='Number of pixels '
                        'to dilate bounding box.')
args = parser.parse_args()
assert args.fill in INPAINTING_FILL_METHODS, \
    f'Fill method must be one of {INPAINTING_FILL_METHODS}.'
######################################## #HERE IS WHERE THE ALPHAMASK CREATION HAPPENS:###########################################

CHEEK_IDXS = OrderedDict([("whole_face", (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,27,26,25,24,23,22,21,20,19,18)),            ###
                          ("left_cheek", (1,17,18,19,20,21,22,23,24,25,26,15,28)),                                              ### 
                          ("right_eye", (36,37,38,39,40,41)),                                                                   ###
                        ("left_eye", (42,43,44,45,46,47)),                                                                      ###
                        ("mouth_edges", (49,50,51,52,53,54,55,56,57,58,59,60))                                                  ###
                                                                                                                                ###
                         ])

                         

detector = dlib.get_frontal_face_detector()                                                                                     ##
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")                                                       ###
                                                                                                                                ###
def imageload():                                                                                                                ###
    # Load image                                                                                                                ##
    #img = "model.png"                                                                                                           ##
    #Create the mask and turn it grey
    showimage()
    img_cv=cv2.imread(filename)
    img2= img_cv
    mask = img2.copy()
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 0)
    for k,d in enumerate(detections):
        shape = predictor(gray, d)
        for (_, name) in enumerate(CHEEK_IDXS.keys()):
            pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32) 
            for i,j in enumerate(CHEEK_IDXS[name]): 
                pts[i] = [shape.part(j).x, shape.part(j).y] 

            pts = pts.reshape((-1,1,2))
            
            #create variables for color and thickness that are black by default
            color = (0, 0, 0)
            thickness = 8

                #turn the impaint areas white

            if name=="mouth_edges":
                color = (200, 200, 200)
            if name=="left_cheek":
                color = (255, 255, 255)
                
                #this is because I can-t code and i dunno how to make the whole image background black so I just set the outline super high
            if name=="whole_face":
                thickness = 1500
                                                                                                                                                      ###
                #creates the Polygons and fills them with color                                                                                      ###
            cv2.fillPoly(mask,[pts],color)                                                                                                          ###
            cv2.polylines(mask,[pts],True,color,thickness)                                                                                         ###

        ##############################################################################################################################################
        
        status = cv2.imwrite('python_output.png',mask)
        

        
        #convert the image to a byte string in PNG format
    


    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
        #Encode the PNG byte string in base64 format
        
    img_b64 = img2b64(im_pil)
    im_pil2 = Image.fromarray(mask)
    mask_b64 = img2b64(im_pil2)
    


    # Run inpainting
    
    #extra_options = {
        #'width': args.width,
        #'height': args.height,
        #'steps': args.steps,
        #'cfg_scale': args.cfg_scale,
        #'sample_name': args.sample_name,
        #'denoising_strength': args.denoising_strength,
        #'mask_blur': args.mask_blur,
        #'inpainting_fill': INPAINTING_FILL_METHODS.index(args.fill),
        #'inpaint_full_res': False
    #}

    request = generate_request(img_b64, prompt=f"(((creative makeup look))), ((beautiful lips)), ((glossy)), ((({variable})))", mask=mask_b64,
                                    negative_prompt="ugly, ((eyes)), (((tear duct))),((Images cut out at the top, left, right, bottom, bad composition mutated body parts)) ((looking down)), (((deformed))), ((pupils)), (((closed eyes)))")#, **extra_options)
    response = submit_post(IMG2IMG_URL, request)
    output_img_b64 = response.json()['images'][0]

        # Save images
    save_encoded_image(output_img_b64, args.output_path)
    mask_path = os.path.join(os.path.dirname(args.output_path),
                            f'mask_{os.path.basename(args.output_path)}')
    save_encoded_image(mask_b64, mask_path)

    img_new = cv2.imread("inpaint-person.png")
    mask_new = cv2.imread("mask_inpaint-person.png")
        
        #search for color in the image
        #masked_img = cv2.bitwise_and(img_new,img_new,mask=mask_new)
        #detecting green colors
    GREEN_MIN = np.array([40, 20, 50],np.uint8)
    GREEN_MAX = np.array([90, 255, 255],np.uint8)
    hsv_img = cv2.cvtColor(img_new,cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)

        #asking if image covers a significant amount of green
    total_pixels = img_new.shape[0] * img_new.shape[1]
    masked_pixels = np.count_nonzero(frame_threshed)
    mask_percentage = masked_pixels / total_pixels * 100

    if mask_percentage > 0.08:
        print("The green mask covers more than 1% of the picture.")
    else:
        print("The green mask does not cover more than 1% of the picture.")


        # Create a window and display the image in it
    #cv2.namedWindow("Image")
    #cv2.imshow("Image", img_new)

        #wait for the user to close the window
    #cv2.waitKey(0)

        #clean up
    #cv2.destroyAllWindows()
    result_image(show)
    product_image(label3, 15)
    product_image(label4, 16)
    product_image(label5, 17)

    
    #reate a window and set title



def imageload_capture():                                                                                                                ###
    # Load image                                                                                                                ##
    #img = "model.png"                                                                                                           ##
    #Create the mask and turn it grey
    
    img_cv=takepicture()
    img2= img_cv
    mask = img2.copy()
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    detections = detector(gray, 0)
    for k,d in enumerate(detections):
        shape = predictor(gray, d)
        for (_, name) in enumerate(CHEEK_IDXS.keys()):
            pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32) 
            for i,j in enumerate(CHEEK_IDXS[name]): 
                pts[i] = [shape.part(j).x, shape.part(j).y] 

            pts = pts.reshape((-1,1,2))
            
            #create variables for color and thickness that are black by default
            color = (0, 0, 0)
            thickness = 8

                #turn the impaint areas white

            if name=="mouth_edges":
                color = (200, 200, 200)
            if name=="left_cheek":
                color = (255, 255, 255)
                
                #this is because I can-t code and i dunno how to make the whole image background black so I just set the outline super high
            if name=="whole_face":
                thickness = 1500
                                                                                                                                                      ###
                #creates the Polygons and fills them with color                                                                                      ###
            cv2.fillPoly(mask,[pts],color)                                                                                                          ###
            cv2.polylines(mask,[pts],True,color,thickness)                                                                                         ###

        ##############################################################################################################################################
        
        status = cv2.imwrite('python_output.png',mask)
        

        
        #convert the image to a byte string in PNG format
    


    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
        #Encode the PNG byte string in base64 format
        
    img_b64 = img2b64(im_pil)
    im_pil2 = Image.fromarray(mask)
    mask_b64 = img2b64(im_pil2)
    


    # Run inpainting
    
    #extra_options = {
        #'width': args.width,
        #'height': args.height,
        #'steps': args.steps,
        #'cfg_scale': args.cfg_scale,
        #'sample_name': args.sample_name,
        #'denoising_strength': args.denoising_strength,
        #'mask_blur': args.mask_blur,
        #'inpainting_fill': INPAINTING_FILL_METHODS.index(args.fill),
        #'inpaint_full_res': False
    #}

    request = generate_request(img_b64, prompt=f"(((creative makeup look))), ((beautiful lips)), ((glossy)), ((({variable})))", mask=mask_b64,
                                    negative_prompt="ugly, ((eyes)), (((tear duct))),((Images cut out at the top, left, right, bottom, bad composition mutated body parts)) ((looking down)), (((deformed))), ((pupils)), (((closed eyes)))")#, **extra_options)
    response = submit_post(IMG2IMG_URL, request)
    output_img_b64 = response.json()['images'][0]

        # Save images
    save_encoded_image(output_img_b64, args.output_path)
    mask_path = os.path.join(os.path.dirname(args.output_path),
                            f'mask_{os.path.basename(args.output_path)}')
    save_encoded_image(mask_b64, mask_path)

    img_new = cv2.imread("inpaint-person.png")
    mask_new = cv2.imread("mask_inpaint-person.png")
        
        #search for color in the image
        #masked_img = cv2.bitwise_and(img_new,img_new,mask=mask_new)
        #detecting green colors
    GREEN_MIN = np.array([40, 20, 50],np.uint8)
    GREEN_MAX = np.array([90, 255, 255],np.uint8)
    hsv_img = cv2.cvtColor(img_new,cv2.COLOR_BGR2HSV)
    frame_threshed = cv2.inRange(hsv_img, GREEN_MIN, GREEN_MAX)

        #asking if image covers a significant amount of green
    total_pixels = img_new.shape[0] * img_new.shape[1]
    masked_pixels = np.count_nonzero(frame_threshed)
    mask_percentage = masked_pixels / total_pixels * 100

    if mask_percentage > 0.08:
        print("The green mask covers more than 1% of the picture.")
    else:
        print("The green mask does not cover more than 1% of the picture.")


        # Create a window and display the image in it
    #cv2.namedWindow("Image")
    #cv2.imshow("Image", img_new)

        #wait for the user to close the window
    #cv2.waitKey(0)

        #clean up
    #cv2.destroyAllWindows()
    result_image(show)
    product_image(label3, 15)
    product_image(label4, 16)
    product_image(label5, 17)

    
    #reate a window and set title




label1 = tk.Label(window, text="choose the vibe of your unique makeup-look: (1. CHOOSE ONE OF THE SMALL BUTTONS FIRST!!!)", font = ('calibri', 14, 'bold'))
label1.pack(pady=5)

button2 = tk.Button(window, fg='black', bg='red', text="clown", command=lambda: [create_variable(button2.cget("text")), update_label_color(button2.cget("text"))])
button3 = tk.Button(window, fg='black', bg='white', text="stripes", command=lambda: [create_variable(button3.cget("text")), update_label_color(button3.cget("text"))])
button4 = tk.Button(window, fg='black', bg='light blue', text="blue", command=lambda: [create_variable(button4.cget("text")), update_label_color(button4.cget("text"))])
button5 = tk.Button(window, fg='black', bg='yellow', text="stars", command=lambda: [create_variable(button5.cget("text")), update_label_color(button5.cget("text"))])
button6 = tk.Button(window, fg='black', bg='coral', text="coral", command=lambda: [create_variable(button6.cget("text")), update_label_color(button6.cget("text"))])
button7 = tk.Button(window, fg='white', bg='blue', text="neon", command=lambda: [create_variable(button7.cget("text")), update_label_color(button7.cget("text"))])
button8 = tk.Button(window, fg='black', bg='white', text="diamonds", command=lambda: [create_variable(button8.cget("text")), update_label_color(button8.cget("text"))])

# Add the buttons to the window
button2.pack(pady=5, padx=5, side=tk.LEFT)
button3.pack(pady=5, padx=5, side=tk.LEFT)
button4.pack(pady=5, padx=5, side=tk.LEFT)
button5.pack(pady=5, padx=5, side=tk.LEFT)
button6.pack(pady=5, padx=5, side=tk.LEFT)
button7.pack(pady=5, padx=5, side=tk.LEFT)
button8.pack(pady=5, padx=5, side=tk.LEFT)
label = tk.Label(window, text="")
label.pack(pady=5)

button1 = tk.Button(window, text="Upload file-->", font=("calibri", 15, "bold", "underline"), bg='pink', command=lambda: imageload())
button1.pack(pady=20, padx=20, side= tk.BOTTOM)

button_image = tk.Button(window, text="Take a picture-->", font=("calibri", 15, "bold", "underline"), bg='aqua', command=lambda: imageload_capture())
button_image.pack(pady=20, padx=20, side= tk.BOTTOM)

#product images
label3 = tk.Label(window)
label3.pack(pady=5)
label4 = tk.Label(window)
label4.pack(pady=5)
label5 = tk.Label(window)
label5.pack(pady=5)

#the result image
show = tk.Label(window)
show.pack(pady=5)



    #Start the main loop of the window
window.mainloop()
#imageload()
        

