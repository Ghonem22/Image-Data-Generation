
import os
import math
import shutil
import glob
import cv2
import matplotlib.pyplot as plt
from random import random, randint

from PIL import Image, ImageOps
import numpy as np
import imutils
import json
from blendmodes.blend import blendLayers, BlendType
from tqdm.auto import tqdm


def box_corner_to_center(boxes):
    """Convert from (upper_left, bottom_right) to (center, width, height)"""
    x1, y1, x2, y2 = boxes[0], boxes[1], boxes[2], boxes[3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = np.stack((cx, cy, w, h), axis=-1)
    return boxes

def rotate_with_transparent_bkground(image, angle):
  """ Take an image and rotate it with a specific given angle and output an image with 
      a transparent background and without cutting in the image"""

  # grab the dimensions of the image and then determine the
  # center
  (h, w) = image.shape[:2]
  (cX, cY) = (w // 2, h // 2)
  # grab the rotation matrix (applying the negative of the
  # angle to rotate clockwise), then grab the sine and cosine
  # (i.e., the rotation components of the matrix)
  M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
  cos = np.abs(M[0, 0])
  sin = np.abs(M[0, 1])
  # compute the new bounding dimensions of the image
  nW = int((h * sin) + (w * cos))
  nH = int((h * cos) + (w * sin))
  # adjust the rotation matrix to take into account translation
  M[0, 2] += (nW / 2) - cX
  M[1, 2] += (nH / 2) - cY

  # perform the actual rotation and return the image
  dst_mat = np.zeros((nH, nW, 4), np.uint8)

  im = cv2.warpAffine(image, M, (nW, nH), dst_mat, flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_TRANSPARENT)
  return im

def Add_Image_To_Background(background_img, added_img, img_dimention, bool_rotate = False):

  # Convert Foreground and background images to RGBA format to include alpha channel 
  added_img = cv2.cvtColor(added_img,cv2.COLOR_BGR2BGRA)
  background_img = cv2.cvtColor(background_img,cv2.COLOR_BGR2BGRA)

  #print(background_img.shape)

  length, width,ch = added_img.shape

  if length > width:
      bool_Vertical = True
    
  else:
      bool_Vertical = False


  if bool_rotate:
    #Generate Angles for rotation 
    theta1 = int(random()* 65)    
    theta2 = 180 + int( random()* 65)
    thetas = [-theta1, theta1, -theta1, theta1, -theta2, theta2]

    #Choose random theta to rotate image with it 
    thata =thetas[int(random()* 6)]
    added_img = rotate_with_transparent_bkground(added_img, thata)
    #print("4 channel values before resizing", added_img[:,:,3])

  #Resize currency image
  Width_to_hight_ratio =np.array(added_img).shape[1] / np.array(added_img).shape[0]
  
  img_dimention = img_dimention + int(random() *10)  # 50
  added_img_dim = (img_dimention, int(img_dimention/Width_to_hight_ratio))

  if bool_Vertical == 1:
      added_img_dim= (int(img_dimention *  Width_to_hight_ratio),img_dimention)
  
  added_img = cv2.resize(added_img, added_img_dim, interpolation = cv2.INTER_AREA)

  #background_dim=(B_D,B_D)
  

  width = 1000 
  height = 700
  
  if bool_Vertical ==0:
      back_dim=(width,height)
  else:
      back_dim=(height,width)

  background_img= cv2.resize(background_img, back_dim, interpolation = cv2.INTER_AREA)

  background_rows, background_cols,channels = background_img.shape
  added_img_rows, added_img_cols, channels = added_img.shape


  num_rest_rows_from_background = abs(background_rows - added_img_rows -10)
  num_rest_cols_from_background = abs(background_cols -added_img_cols -10)

  #x1 and y1 is the points of the upper left corner of the image 
  y1 =  randint(0,num_rest_rows_from_background)              #  int(range(num_rest_rows_from_background)[int(random()* num_rest_rows_from_background)] * 1)
  x1 = randint(0,num_rest_cols_from_background)               #   int(range(num_rest_cols_from_background)[int(random()* num_rest_cols_from_background)] * 1)

  added_img = cv2.GaussianBlur(added_img,(5,5),0)

  background = Image.fromarray(background_img[y1:added_img_rows+y1, x1:added_img_cols+x1])
  added_img = Image.fromarray(added_img)

  
  out_im = blendLayers(background, added_img, BlendType.GLOW)
  out_im = out_im.convert('RGB')

  out_im = np.array(out_im)
  added_img = np.array(added_img)
  background_img = cv2.cvtColor(background_img,cv2.COLOR_BGRA2BGR)

  background_img[y1:added_img_rows+y1, x1:added_img_cols+x1]= out_im

  #print(added_img.shape)
  #print(background_img.shape)
  
  # Add our image to the background image, so the background image become our added image + background
  #

  # Change Image Brightness 
 
  aa= int(random() *25)- int(random() *25)
  background_img = background_img +  [aa + int(random() *10)- int(random() *10),
                aa + int(random() *10)- int(random() *10),
                aa + int(random() *10)- int(random() *10)] 
 
  

  # calculating currency dtection parameters:
  x2 = x1 + added_img_cols 
  y2 = y1 + added_img_rows 
  boxes = [x1, y1, x2, y2] 

  box = box_corner_to_center(boxes)
  X_center, Y_center, X_len, Y_len = box

  fianl_img_Height, fianl_img_Width, _ = background_img.shape

  X_center = X_center / fianl_img_Width
  Y_center = Y_center / fianl_img_Height
  X_len = X_len / fianl_img_Width
  Y_len = Y_len / fianl_img_Height
  

  parameters =str(X_center) + " " +  str(Y_center) + " " + str(X_len)+ " " + str(Y_len)
  

  return background_img , parameters



def itrate_back(added_img, background_imgs_dir, Saving_path_for_each_Data_Class, img_dimention, i1, i2, label, Data_Class,  bool_rotate):  
  """ Iterate over background images and save new images and text files """
  """ Input parameters: 
      added_img : The image added to the bakground
      background_imgs_dir: The directory of all images used to be background 
      Saving_path : path of saved files 
      img_dimention: Dimention needed to resize added image 
      label : The class number 
      bool_rotate : If its value is True a rotation will be made to the image  

       """
  data_path = os.path.join(background_imgs_dir,'*g')
  files = glob.glob(data_path) # Paths of all background images 
  for i, im_path in enumerate(files[i1:i2]):
    #print(im_path)
    background_img = cv2.imread(im_path, -1) # this is the background image 
    #img = cv2.cvtColor(img,cv2.COLOR_BGR2BGRA) 

    #print("iterate", img.shape)
    out_img, parameters = Add_Image_To_Background(background_img, added_img, img_dimention, bool_rotate)
    #out = cv2.GaussianBlur(out,(5,5),0)

    #write parameters into text file

    new_img_name = Saving_path_for_each_Data_Class +'/' + str(Data_Class)  + '_' + str(i1) + str(int(random() *10**6))
    img_saving_directory = new_img_name + ".jpg"
    
    txt_saving_directory = new_img_name + ".txt"
    
    with open(txt_saving_directory, 'w') as f:
        f.write(str(label) + " "+ parameters)
        # Save label Box parameters "dimentions" needed for YOLO training which is 
        # Label X_center Y_center Width Hight 
        f.close()
          
    
    cv2.imwrite(img_saving_directory, out_img) # save image in the director 

def Make_DataSet(Data_Class_images_path, Different_Background_starting_points,label,the_increase, img_dir, dimention_ranges,
                 Saving_path_for_each_Data_Class, num_of_images_for_each_Data_Class, Data_Class,  bool_rotate = False):

  """ this function take each Data_Class image and put it on different background
        with different sizes 
  """
  # i1 and i2 define the reange of background images that each Data_Class image iterate on 

  num_of_generated_images = 0               # use it to break loop if it excceds  "num_of_images_for_each_Data_Class"

  for i, start_pt in enumerate(tqdm(Different_Background_starting_points)):
    path = Data_Class_images_path[i] 

    added_img = cv2.imread(path, -1) # read added "Data_Class" image 
    for img_dimention in dimention_ranges:

        if num_of_generated_images >=  num_of_images_for_each_Data_Class:
            break
        
        i1 = int(start_pt)
        i2 = int(start_pt +  the_increase)

        
        try:
            itrate_back(added_img, img_dir,Saving_path_for_each_Data_Class,img_dimention,i1,i2,label, Data_Class, bool_rotate)
        except:
            print("error appear when adding Data_Class_ image: ", path, "to a background image")
        
        start_pt += the_increase

        num_of_generated_images += i2 - i1




# Generate Full DataSet

def generate_dataset(original_Data_Class_images_path, general_images_path, saving_path, dimention_ranges,
                     num_of_images_for_each_Data_Class = 500, starting_Label = 0, starting_point_for_Bg_Image = 0):


    # get the names of the new classes
    Data_Classes_names = [x[1] for x in os.walk(original_Data_Class_images_path)][0]

    num_of_new_Data_Classes = len(Data_Classes_names)

    #Create Saving folders for the generated data carry the name of Data_Classes

    saving_folder_path = os.path.join(saving_path, 'Generated_Data')
    
    # get num of old Data_Classes inside the folder
    if  os.path.exists(saving_folder_path):
        old_Data_Classes_names = [x[1] for x in os.walk(saving_folder_path)][0]
        num_of_old_Data_Classes = len(old_Data_Classes_names)
        
    else:
        old_Data_Classes_names = []
        num_of_old_Data_Classes = 0

    if not os.path.exists(saving_folder_path):
        os.mkdir(saving_folder_path)

    #create subfolders with the name of new Data_Classes

    existed_Data_Classes = []

    
    # get the 'starting_Label' and 'starting_point_for_Bg_Image' from the previous process

    try:
        previous = open(os.path.join(saving_folder_path, 'file.txt'),"r")
        previous_data = json.loads(previous.read())

        try:
            starting_Label = previous_data['starting_label'] 

        # if file not contain starting_Label    
            
        except:
            starting_Label = starting_Label 
            
        try:
            starting_point_for_Bg_Image = previous_data['starting_point_for_Bg_Image']  + 1

        # if file not contain starting_point_for_Bg_Image
        except:
            starting_point_for_Bg_Image = starting_point_for_Bg_Image
    except:
        starting_Label = starting_Label
        starting_point_for_Bg_Image = starting_point_for_Bg_Image
        
    # get number of background images
    bg_data_path = os.path.join(general_images_path,'*g')
    bg_images_path = glob.glob(bg_data_path)
    num_of_bg_img = len(bg_images_path) # number of Data_Classes images we need

    # Generate Data for each Data_Class
    all_generated_images_path = []
    for Data_Class in Data_Classes_names: 

        try:
            path = os.path.join(saving_folder_path, Data_Class)
            os.mkdir(path)
        except:
            # continue if there is no empty folder with the name of this Data_Class
            if len(os.listdir(path)) > num_of_images_for_each_Data_Class -3:   # 3 is just a marjin
                existed_Data_Classes.append(Data_Class)
                print("folder alerady exist with the name of ", Data_Class)
                continue


        Saving_path_for_each_Data_Class = os.path.join(saving_folder_path, Data_Class)    

        img_Data_Class_path = os.path.join(original_Data_Class_images_path, Data_Class)    

        data_path = os.path.join(img_Data_Class_path,'*g')
        Data_Class_images_path = glob.glob(data_path)
        num_of_added_img = len(Data_Class_images_path) # number of Data_Classes images we need

        # n: is the number of generated images for each added "Data_Class" image 

        if num_of_added_img == 0:
            print("there is no Data_Class images to use")
            continue
            
        num_generated_img_for_each_img = math.ceil(num_of_images_for_each_Data_Class / num_of_added_img)
            
        if num_generated_img_for_each_img < 1:
            num_generated_img_for_each_img = 1
            
        """
        Background_starting_points : if we have 10 images for a Data_Class and we need 100 generated image for that Data_Class 
        so we need to generate 10 images for each Data_Class image, assume we start from image 
        400 in our background images, so the values in this list will be 
        [400, 410, 420, 430, ...], so we make sure that we will take 10 different 
        background images for each Data_Class 
        """ 
        Background_starting_points = [starting_point_for_Bg_Image + j*num_generated_img_for_each_img \
                                                for j in range(num_of_added_img)]

        # dimention_ranges = [80 ,120,190, 270, 360] # dimention of added 'Data_Class' image 
        """the_increase: the number of images for each dimention, so if we have 10 generated image 
        for the first Data_Class image, we will have 2 images for each dimention 
        """

        the_increase = math.ceil(num_generated_img_for_each_img / len(dimention_ranges))
        print(f"Loading Class {Data_Class}")
        Make_DataSet(Data_Class_images_path, Background_starting_points, starting_Label, the_increase, general_images_path, dimention_ranges, Saving_path_for_each_Data_Class, num_of_images_for_each_Data_Class, Data_Class)


        generated_data_path = os.path.join(Saving_path_for_each_Data_Class,'*g')
        generated_images_path = glob.glob(generated_data_path)
        all_generated_images_path.extend(generated_images_path)
        
        starting_Label += 1
        starting_point_for_Bg_Image += num_of_images_for_each_Data_Class 
        if starting_point_for_Bg_Image >= num_of_bg_img:
            starting_point_for_Bg_Image = 0
        print(f"Class {Data_Class} Completed\n")
    
    # write paths of all generated images in txt file


    with open( os.path.join(saving_folder_path, 'generated_images_path.txt'), 'w') as f:       
        f.write("\n".join(all_generated_images_path))

    # get names of all generated Data_Classes // removing repeated
    new_generated_Data_Classes = [x for x in Data_Classes_names if x not in existed_Data_Classes] 

    with open( os.path.join(saving_folder_path, 'new_Data_Classes_names.txt'), 'w') as f:       
        f.write("\n".join(new_generated_Data_Classes))



    # Save txt file with thw last 
    exDict = {'starting_label': starting_Label  , 'starting_point_for_Bg_Image': starting_point_for_Bg_Image }


    with open(os.path.join(saving_folder_path, 'file.txt'), 'w') as file:
        file.write(json.dumps(exDict)) 
        
    print("Completed sucssefully, Congrate :)")

    with open( os.path.join(saving_folder_path, 'names.txt'), 'a') as f:       
        for item in Data_Classes_names:
            if item in old_Data_Classes_names:
                continue
            f.write("%s\n" % item)

    num_of_current_Data_Classes = len([x[1] for x in os.walk(saving_folder_path)][0])
    
    if num_of_current_Data_Classes != num_of_new_Data_Classes + num_of_old_Data_Classes:
        repeated = num_of_new_Data_Classes + num_of_old_Data_Classes - num_of_current_Data_Classes
        print(f"there are {repeated} Data_Classes repeated: {existed_Data_Classes}: \n If these Data_Classes are very diffrent from the existed ones, plz give them diiffrent names")

    return existed_Data_Classes, num_of_old_Data_Classes, num_of_new_Data_Classes

