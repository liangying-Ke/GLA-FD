import os
import glob
import random
import pickle
import configs
from tqdm import tqdm


def create_FVUSM_annotation(args):
	def iter(root, train_samples=[], test_samples=[], sub2classes={}, phase='train'):
		for sub in tqdm(os.listdir(root)):
			if not os.path.isdir(os.path.join(root, sub)):
				continue
			paths = glob.glob(os.path.join(root, sub, '*.jpg'))
			random.shuffle(paths)

			train, test = args.split.split(':')
			ratio = int(test) / (int(train) + int(test))
			bps = int(len(paths) * ratio)
			for_test = paths[:bps]

			if sub not in sub2classes:
				sub2classes[sub] = len(sub2classes)

			for path in paths:
				if path in for_test:
					test_samples.append({'path':path, 'label':sub2classes[sub]})
				else:
					train_samples.append({'path':path, 'label':sub2classes[sub]})

		return train_samples, test_samples, sub2classes
	
	train_samples, test_samples, sub2classes = iter(os.path.join(args.data_root, '1st_session', 'extractedvein'))
	train_samples, test_samples, sub2classes = iter(os.path.join(args.data_root, '2nd_session', 'extractedvein'), train_samples, test_samples, sub2classes)
	pickle.dump({
		'train_set':train_samples, 
		'test_set':test_samples,
	}, open(args.annot_file, 'wb'))
	print(f'train_samples: {len(train_samples)}')
	print(f'test_samples: {len(test_samples)}')
	print(test_samples[0])


def create_PLUSVein_annotation(args):
	def iter(root, Where):
		sub2classes = {}
		train_samples, test_samples = [], []
		data_path = os.path.join(root, Where)

		for folder in tqdm(os.listdir(data_path)):
			if not os.path.isdir(os.path.join(data_path, folder)):
				continue
			paths = glob.glob(os.path.join(data_path, folder, '*.png'))
			for idx in ['02', '03', '04', '07', '08', '09']:
				identity = f'{folder}_{idx}'
				filter_paths = [path for path in paths if identity in path]
				random.shuffle(filter_paths)
				
				train, test = args.split.split(':')
				ratio = int(test) / (int(train) + int(test))
				bps = int(len(filter_paths) * ratio)
				for_test = filter_paths[:bps]

				if not identity in sub2classes:
					sub2classes[identity] = len(sub2classes)
			
				for path in filter_paths:
					if path in for_test:
						test_samples.append({'path':path, 'label':sub2classes[identity]})
					else:
						train_samples.append({'path':path, 'label':sub2classes[identity]})
		return train_samples, test_samples
	

	train_samples_LED, test_samples_LED = iter(args.data_root, os.path.join('PLUS-FV3-LED', 'PALMAR', '01'))
	train_samples_LASER, test_samples_LASER = iter(args.data_root, os.path.join('PLUS-FV3-Laser', 'PALMAR', '01'))
	pickle.dump({
		'LED':{
			'train_set':train_samples_LED, 
			'test_set':test_samples_LED,
		},
		'LASER':{
			'train_set':train_samples_LASER, 
			'test_set':test_samples_LASER,
		}
	}, open(args.annot_file, 'wb'))
	print(f'train_samples: {len(train_samples_LED)}')
	print(f'test_samples: {len(test_samples_LED)}')
	print(test_samples_LED[0])



def create_MMCBNU_annotation(args):
	def iter(root_path):
		sub2classes = {}
		train_samples, test_samples = [], []

		for sub in tqdm(os.listdir(root_path)):
			sub_path = os.path.join(root_path, sub)

			if not os.path.isdir(sub_path):
				continue

			for finger in os.listdir(sub_path):
				finger_path = os.path.join(sub_path, finger)

				if not os.path.isdir(finger_path):
					continue

				paths = glob.glob(os.path.join(finger_path, '*.bmp'))
				
				random.shuffle(paths)
				
				train, test = args.split.split(':')
				ratio = int(test) / (int(train) + int(test))
				bps = int(len(paths) * ratio)
				for_test = paths[:bps]

				if not finger_path in sub2classes:
					sub2classes[finger_path] = len(sub2classes)

				for path in paths:
					if path in for_test:
						test_samples.append({'path':path, 'label':sub2classes[finger_path]})
					else:
						train_samples.append({'path':path, 'label':sub2classes[finger_path]})
		return train_samples, test_samples
	
	train_samples, test_samples = iter(args.data_root)
	pickle.dump({
		'train_set':train_samples, 
		'test_set':test_samples,
	}, open(args.annot_file, 'wb'))

	print(f'train_samples: {len(train_samples)}')
	print(f'test_samples: {len(test_samples)}')
	print(train_samples[0])



def create_UTFVP_annotation(args):
	def iter(root_path):
		sub2classes = {}
		train_samples, test_samples = [], []

		for sub in tqdm(os.listdir(root_path)):
			sub_path = os.path.join(root_path, sub)

			if not os.path.isdir(sub_path):
				continue
			
			for finger in range(1, 7):
				paths = glob.glob(os.path.join(sub_path, f'{sub}_{finger}_*.png'))
				
				random.shuffle(paths)
				
				train, test = args.split.split(':')
				ratio = int(test) / (int(train) + int(test))
				bps = int(len(paths) * ratio)
				for_test = paths[:bps]

				if not f'{sub}_{finger}' in sub2classes:
					sub2classes[f'{sub}_{finger}'] = len(sub2classes)

				for path in paths:
					if path in for_test:
						test_samples.append({'path':path, 'label':sub2classes[f'{sub}_{finger}']})
					else:
						train_samples.append({'path':path, 'label':sub2classes[f'{sub}_{finger}']})

		return train_samples, test_samples
	
	train_samples, test_samples = iter(args.data_root)
	pickle.dump({
		'train_set':train_samples, 
		'test_set':test_samples,
	}, open(args.annot_file, 'wb'))

	print(f'train_samples: {len(train_samples)}')
	print(f'test_samples: {len(test_samples)}')
	print(train_samples[0])


def create_NUPT_annotation(args):
	def iter(root_path, train_samples=[], test_samples=[], sub2classes={}):

		for sub in tqdm(os.listdir(root_path)):
			sub_path = os.path.join(root_path, sub)

			if not os.path.isdir(sub_path):
				continue
			
			paths = glob.glob(os.path.join(sub_path, f'*.bmp'))
			
			random.shuffle(paths)
			
			train, test = args.split.split(':')
			ratio = int(test) / (int(train) + int(test))
			bps = int(len(paths) * ratio)
			for_test = paths[:bps]

			if not sub in sub2classes:
				sub2classes[sub] = len(sub2classes)

			for path in paths:
				if path in for_test:
					test_samples.append({'path':path, 'label':sub2classes[sub]})
				else:
					train_samples.append({'path':path, 'label':sub2classes[sub]})

		return train_samples, test_samples, sub2classes
	
	
	train_samples, test_samples, sub2classes = iter(os.path.join(args.data_root, 'Process_gray_full_840_1class', 'FV_process_gray_1class'))
	train_samples, test_samples, sub2classes = iter(os.path.join(args.data_root, 'Process_gray_full_840_2class', 'FV_process_gray_2class'), train_samples, test_samples, sub2classes)
	pickle.dump({
		'train_set':train_samples, 
		'test_set':test_samples,
	}, open(args.annot_file, 'wb'))

	print(f'train_samples: {len(train_samples)}')
	print(f'test_samples: {len(test_samples)}')
	print(train_samples[0])


if __name__== '__main__':
	args = configs.get_all_params()
	
	args.datasets = 'FV-USM'
	args = configs.get_dataset_params(args)
	create_FVUSM_annotation(args)
	
	args.datasets = 'PLUSVein-FV3'
	args = configs.get_dataset_params(args)
	create_PLUSVein_annotation(args)
	
	args.datasets = 'MMCBNU_6000'
	args = configs.get_dataset_params(args)
	create_MMCBNU_annotation(args)
	
	args.datasets = 'UTFVP'
	args = configs.get_dataset_params(args)
	create_UTFVP_annotation(args)
	
	args.datasets = 'NUPT-FPV'
	args = configs.get_dataset_params(args)
	create_NUPT_annotation(args)
	
