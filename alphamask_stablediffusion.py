from collections import OrderedDict
import numpy as np
import cv2
import dlib
import imutils

import os
import json
import base64
import io
import requests

import torch
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision import transforms
from torchvision.io.image import read_image
import base64
from PIL import Image


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


INPAINTING_FILL_METHODS = ['fill', 'original', 'latent_noise', 'latent_nothing']




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inpaint instances of people using stable '
                                                 'diffusion.')
    parser.add_argument('img_path', type=str, help='Path to input image.')
    parser.add_argument('-o', '--output_path', type=str, default='inpaint-person.png',
                        help='Path to output image.')
    parser.add_argument('-p', '--prompt', type=str, default='',
                        help='Stable diffusion prompt to use.')
    parser.add_argument('-n', '--negative_prompt', type=str, default='person',
                        help='Stable diffusion negative prompt.')
    parser.add_argument('-W', '--width', type=int, default=768, help='Width of output image.')
    parser.add_argument('-H', '--height', type=int, default=768, help='Height of output image.')
    parser.add_argument('-s', '--steps', type=int, default=30, help='Number of diffusion steps.')
    parser.add_argument('-c', '--cfg_scale', type=int, default=8, help='Classifier free guidance '
                        'scale, i.e. how strongly the image should conform to prompt.')
    parser.add_argument('-S', '--sample_name', type=str, default='Euler a', help='Name of sampler '
                        'to use.')
    parser.add_argument('-d', '--denoising_strength', type=float, default=0.75, help='How much to '
                        'disregard original image.')
    parser.add_argument('-f', '--fill', type=str, default=INPAINTING_FILL_METHODS[0],
                        help='The fill method to use for inpainting.')
    parser.add_argument('-b', '--mask_blur', type=int, default=8, help='Blur radius of Gaussian '
                        'filter to apply to mask.')
    parser.add_argument('-B', '--bounding_box', action='store_true', help='Convert mask to '
                        'bounding box.')
    parser.add_argument('-D', '--bbox_dilation', type=float, default=16, help='Number of pixels '
                        'to dilate bounding box.')
    args = parser.parse_args()
    assert args.fill in INPAINTING_FILL_METHODS, \
        f'Fill method must be one of {INPAINTING_FILL_METHODS}.'

    CHEEK_IDXS = OrderedDict([("whole_face", (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,27,26,25,24,23,22,21,20,19,18)),
                          ("left_cheek", (1,17,18,19,20,21,22,23,24,25,26,15,28)),
                          ("right_eye", (36,37,38,39,40,41)),
                        ("left_eye", (42,43,44,45,46,47)),
                        ("right_cheek", (49,50,51,52,53,54,55,56,57,58,59,60))
                        
                         ])

                         

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Load image
    img = cv2.imread(args.img_path)


    mask = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    detections = detector(gray, 0)
    for k,d in enumerate(detections):
        shape = predictor(gray, d)
        for (_, name) in enumerate(CHEEK_IDXS.keys()):
            pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32) 
            for i,j in enumerate(CHEEK_IDXS[name]): 
                pts[i] = [shape.part(j).x, shape.part(j).y]
        
            pts = pts.reshape((-1,1,2))
            color = (0, 0, 0)
            thickness = 10
            if name=="right_cheek":
                color = (255, 255, 255)
            if name=="left_cheek":
                color = (255, 255, 255)
            if name=="whole_face":
                thickness = 1000

        
            cv2.fillPoly(mask,[pts],color)
            cv2.polylines(mask,[pts],True,color,thickness)
        
    
        cv2.imshow("Image", mask)
        status = cv2.imwrite('python_output.png',mask)
        print(name)
        cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    


    # Read the image using cv2
    
    # Convert the image to a byte string in PNG format
  
    


    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
# Encode the PNG byte string in base64 format
    
    img_b64 = img2b64(im_pil)
    im_pil2 = Image.fromarray(mask)
    mask_b64 = img2b64(im_pil2)


    # Run inpainting

    extra_options = {
        'width': args.width,
        'height': args.height,
        'steps': args.steps,
        'cfg_scale': args.cfg_scale,
        'sample_name': args.sample_name,
        'denoising_strength': args.denoising_strength,
        'mask_blur': args.mask_blur,
        'inpainting_fill': INPAINTING_FILL_METHODS.index(args.fill),
        'inpaint_full_res': False
    }
    request = generate_request(img_b64, prompt=args.prompt, mask=mask_b64,
                                negative_prompt="ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft, grainy", **extra_options)
    response = submit_post(IMG2IMG_URL, request)
    output_img_b64 = response.json()['images'][0]

    # Save images
    save_encoded_image(output_img_b64, args.output_path)
    mask_path = os.path.join(os.path.dirname(args.output_path),
                             f'mask_{os.path.basename(args.output_path)}')
    save_encoded_image(mask_b64, mask_path)

