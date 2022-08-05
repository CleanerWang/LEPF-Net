import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob

def populate_train_list(orig_images_path, hazy_images_path):

	train_list = []
	tmp_dict = {}
	train_keys = []
	
	image_list_haze = glob.glob(hazy_images_path + "*.*")

	for image in image_list_haze:
		
		image = image.split("/")[-1]
		key = image.split("_")[0] + "_" + image.split("_")[1] + "." + image.split(".")[-1]
	
   
		if key in tmp_dict.keys():
			tmp_dict[key].append(image)
		else:
			tmp_dict[key] = []
			tmp_dict[key].append(image)

	len_keys = len(tmp_dict.keys())
	for i in range(len_keys):
		train_keys.append(list(tmp_dict.keys())[i])
		
	for key in list(tmp_dict.keys()):
		for hazy_image in tmp_dict[key]:
			train_list.append([orig_images_path + key, hazy_images_path + hazy_image])

	return train_list

	

class loader(data.Dataset):

	def __init__(self, orig_images_path, hazy_images_path, is_UWCNN_flag):
		self.isUWCNN_Set = is_UWCNN_flag
		self.data_list = populate_train_list(orig_images_path, hazy_images_path) 
		if is_UWCNN_flag !=1:
			print("Total training examples:", len(self.data_list))

	def __getitem__(self, index):

		data_orig_path, data_hazy_path = self.data_list[index]

		if self.isUWCNN_Set != 1 :
			data_orig = Image.open(data_orig_path)
			data_hazy = Image.open(data_hazy_path)

			data_orig = data_orig.resize((350,350), Image.ANTIALIAS)
			data_hazy = data_hazy.resize((350,350), Image.ANTIALIAS)

			data_orig = (np.asarray(data_orig)/255.0)
			data_hazy = (np.asarray(data_hazy)/255.0)

			data_orig = torch.from_numpy(data_orig).float()
			data_hazy = torch.from_numpy(data_hazy).float()

			return data_orig.permute(2,0,1), data_hazy.permute(2,0,1)
		else:
			return data_orig_path, data_hazy_path

	def __len__(self):
		return len(self.data_list)


