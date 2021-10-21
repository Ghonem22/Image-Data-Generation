# Image-Data-Generation
Image Data Generation  will help you when you suffer from data scarcity, with few images you will be able to build a dataset with hundred of thousands of images

## What will you need as input of the Generation Fucnction:

  1- few cropped for each class: each of these images will be itrated over mutiple images afreter applying some processing (spatial and intensity variation). it's "Egyptian Currency" folder here.
  
  2- Background images: it's the images which our cropped images will be itrated over it, it can be any kind of images, it will be just a background. we downloaded and unzipped Coco dataset to use it as background images.
  
  3- saving_path: it's the path the we will create Dataset folder on it.
  
  4- dimention_ranges: it's ranges of dimentions of the images of the classes, the larger, the larger will be the image of the class compared to its background. it's a list and we can add any number of dimentions, ex: dimention_ranges = [100 ,120,190, 270, 360, 380, 415,500] 

  5- starting_Label: it's the label of the class which we will write in the txt file, we will use that later in detection.
  
  6- starting_point_for_Bg_Image: the starting image from the background dataset, we use it when we want to add an additional class without using repated background images.



## Function output:

  Folder "Generated_Data" that contains Data and some useful txt files:
  
   1- n folders carry the names of the classes, each folder includes class images, and txt file for each image that determine object location and label.
    
   2- text file called "name.txt" contains the names of all classes.

   3- text file called "file.txt", this is an example for what it contains: {"starting_label": 12, "starting_point_for_Bg_Image": 4800}. These information is useful when we eant to add additional class to out dataset without any conflict.
    
   4- text file called "generated_images_path.txt" that contains path of all images.
    
   5- text file called "new_classes_names.txt" that contains names of new classes, which wasn't existed when we generated the dataset. this is useful when we want to add some classed to our dataset, it will add new classed only and write their names into this text file .


## What will you get with this function:

   1- you will get a complete dataset with any quantity you want, then you can use that in classification or detection.
   
   2- the output model won't be ideal when you try it with realistic data, it maybe good enough in some cases, but you need one additional step to make sure it will be perfect: 

   use the output model as pretrained model with a small realistic dataset, in this case you will need a dataset, but you will get a great result with very small dataset.
        
