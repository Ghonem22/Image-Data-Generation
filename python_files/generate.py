from utilis import *
import argparse

def main():
    parser = argparse.ArgumentParser(
        'Genertae',
        description='Genertate dataset')
    parser.add_argument('-l', '--logo_images',
                        help='path of the original images that we want to make dataset for them',
                        required=False,
                        dest='logo_images',
						default = 'logo_images')

    parser.add_argument('-b', '--background_images',
                        help='the background images',
                        required=False,
                        dest='background_images',
                        default='train2014')

    parser.add_argument('-s', '--saving_path',
                        help='tha directory where we will save resutls',
                        required=False,
                        dest='saving_path',
                        default=os.getcwd())


    parser.add_argument('-n', '--num_of_images_for_each_class',
                        help='num_of_images_for_each_class',
                        required=False,
                        type=int,
                        dest='num_of_images_for_each_class',
                        default=300)


    parser.add_argument('-r', '--dimention_ranges',
                        help='the reange of the result image dimentions',
                        required=False,
                        type=list,
                        dest='dimention_ranges',
                        default= [120,190, 270, 360, 380, 415,500] )


    args = parser.parse_args()
	
    x,y,z = generate_dataset(args.logo_images, args.background_images, args.saving_path, args.dimention_ranges, args.num_of_images_for_each_class)



if __name__ == '__main__':
    main()


