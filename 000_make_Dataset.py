import os
import glob
import random
import pickle
import configs
from tqdm import tqdm


def create_FVUSM_annotation(args):
	def iter(root, train_samples=[], val_samples=[], test_samples=[], sub2classes={}, phase='train'):
		for sub in tqdm(os.listdir(root)):
			if not os.path.isdir(os.path.join(root, sub)):
				continue
			paths = glob.glob(os.path.join(root, sub, '*.jpg'))
			random.shuffle(paths)
			breakpoints = int(len(paths) * args.split)
			for_val = paths[:breakpoints]
			for_test = paths[breakpoints:breakpoints*2]
			if sub not in sub2classes:
				sub2classes[sub] = len(sub2classes)

			for path in paths:
				if path in for_val:
					val_samples.append({'path':path, 'label':sub2classes[sub]})
				elif path in for_test:
					test_samples.append({'path':path, 'label':sub2classes[sub]})
				else:
					train_samples.append({'path':path, 'label':sub2classes[sub]})

		return train_samples, val_samples, test_samples, sub2classes
	
	trainingSamples, validatingSamples, testingSamples, sub2classes = iter(os.path.join(args.data_root, '1st_session', 'extractedvein'))
	trainingSamples, validatingSamples, testingSamples, sub2classes = iter(os.path.join(args.data_root, '2nd_session', 'extractedvein'), trainingSamples, validatingSamples, testingSamples, sub2classes)
	pickle.dump({
		'Training_Set':trainingSamples, 
		'Validating_Set':validatingSamples, 
		'Testing_Set':testingSamples,
	}, open(args.annot_file, 'wb'))
	print(f'train_samples: {len(trainingSamples)}')
	print(f'val_samples: {len(validatingSamples)}')
	print(f'test_samples: {len(testingSamples)}')
	print(testingSamples[0])


def create_PLUSVein_annotation(args):
	def iter(root, Where):
		sub2classes = {}
		trainingSamples, validatingSamples, testingSamples = [], [], []
		data_path = os.path.join(root, Where)

		for folder in tqdm(os.listdir(data_path)):
			if not os.path.isdir(os.path.join(data_path, folder)):
				continue
			paths = glob.glob(os.path.join(data_path, folder, '*.png'))
			for idx in ['02', '03', '04', '07', '08', '09']:
				identity = f'{folder}_{idx}'
				filter_paths = [path for path in paths if identity in path]
				random.shuffle(filter_paths)
				breakpoints = int(len(filter_paths) * args.split)
				forTest = filter_paths[:breakpoints]
				forValidate = filter_paths[breakpoints:breakpoints*2]

				if not identity in sub2classes:
					sub2classes[identity] = len(sub2classes)
			
				for path in filter_paths:
					if path in forValidate:
						validatingSamples.append({'path':path, 'label':sub2classes[identity]})
					elif path in forTest:
						testingSamples.append({'path':path, 'label':sub2classes[identity]})
					else:
						trainingSamples.append({'path':path, 'label':sub2classes[identity]})
		return trainingSamples, validatingSamples, testingSamples
	
	trainingSamples_LED, validatingSamples_LED, testingSamples_LED = iter(
		args.data_root, os.path.join('PLUS-FV3-LED', 'PALMAR', '01'))
	trainingSamples_LASER, validatingSamples_LASER, testingSamples_LASER = iter(
		args.data_root, os.path.join('PLUS-FV3-Laser', 'PALMAR', '01'))
	pickle.dump({
		'LED':{
			'Training_Set':trainingSamples_LED, 
			'Validating_Set':validatingSamples_LED, 
			'Testing_Set':testingSamples_LED,
		},
		'LASER':{
			'Training_Set':trainingSamples_LASER, 
			'Validating_Set':validatingSamples_LASER, 
			'Testing_Set':testingSamples_LASER,
		}
	}, open(args.annot_file, 'wb'))
	print(f'train_samples: {len(trainingSamples_LED)}')
	print(f'val_samples: {len(validatingSamples_LED)}')
	print(f'test_samples: {len(testingSamples_LED)}')
	print(testingSamples_LASER[0])



def create_MMCBNU_annotation(args):
	def iter(root_path):
		sub2classes = {}
		training_samples, validating_samples, testing_samples = [], [], []

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
				breakpoints = int(len(paths) * args.split)
				forTest = paths[:breakpoints]
				forValidate = paths[breakpoints:breakpoints*2]
				
				if not finger_path in sub2classes:
					sub2classes[finger_path] = len(sub2classes)
				for path in paths:
					if path in forValidate:
						validating_samples.append({'path':path, 'label':sub2classes[finger_path]})
					elif path in forTest:
						testing_samples.append({'path':path, 'label':sub2classes[finger_path]})
					else:
						training_samples.append({'path':path, 'label':sub2classes[finger_path]})
		return training_samples, validating_samples, testing_samples
	
	training_samples, validating_samples, testing_samples = iter(args.data_root)
	pickle.dump({
		'Training_Set':training_samples, 
		'Validating_Set':validating_samples, 
		'Testing_Set':testing_samples,
	}, open(args.annot_file, 'wb'))

	print(f'train_samples: {len(training_samples)}')
	print(f'val_samples: {len(validating_samples)}')
	print(f'test_samples: {len(testing_samples)}')
	print(training_samples[0])



def create_UTFVP_annotation(args):
	def iter(root_path):
		sub2classes = {}
		training_samples, validating_samples, testing_samples = [], [], []

		for sub in tqdm(os.listdir(root_path)):
			sub_path = os.path.join(root_path, sub)

			if not os.path.isdir(sub_path):
				continue
			
			for finger in range(1, 7):
				paths = glob.glob(os.path.join(sub_path, f'{sub}_{finger}_*.png'))
				
				random.shuffle(paths)
				breakpoints = int(len(paths) * args.split)
				forTest = paths[:breakpoints]
				
				if not f'{sub}_{finger}' in sub2classes:
					sub2classes[f'{sub}_{finger}'] = len(sub2classes)
				for path in paths:
					if path in forTest:
						testing_samples.append({'path':path, 'label':sub2classes[f'{sub}_{finger}']})
						validating_samples.append({'path':path, 'label':sub2classes[f'{sub}_{finger}']})
					else:
						training_samples.append({'path':path, 'label':sub2classes[f'{sub}_{finger}']})

		return training_samples, validating_samples, testing_samples
	
	training_samples, validating_samples, testing_samples = iter(args.data_root)
	pickle.dump({
		'Training_Set':training_samples, 
		'Validating_Set':validating_samples, 
		'Testing_Set':testing_samples,
	}, open(args.annot_file, 'wb'))

	print(f'train_samples: {len(training_samples)}')
	print(f'val_samples: {len(validating_samples)}')
	print(f'test_samples: {len(testing_samples)}')
	print(training_samples[0])


def create_NUPT_annotation(args):
	def iter(root_path, training_samples=[], validating_samples=[], testing_samples=[], sub2classes={}):

		for sub in tqdm(os.listdir(root_path)):
			sub_path = os.path.join(root_path, sub)

			if not os.path.isdir(sub_path):
				continue
			
			paths = glob.glob(os.path.join(sub_path, f'*.bmp'))
			
			random.shuffle(paths)
			forTrain = paths[:4]
			forVal = paths[4:5]
			
			if not sub in sub2classes:
				sub2classes[sub] = len(sub2classes)

			for path in paths:
				if path in forVal:
					validating_samples.append({'path':path, 'label':sub2classes[sub]})
				elif path in forTrain:
					training_samples.append({'path':path, 'label':sub2classes[sub]})
				else:
					testing_samples.append({'path':path, 'label':sub2classes[sub]})

		return training_samples, validating_samples, testing_samples, sub2classes
	
	
	training_samples, validating_samples, testing_samples, sub2classes = iter(os.path.join(args.data_root, 'Process_gray_full_840_1class', 'FV_process_gray_1class'))
	training_samples, validating_samples, testing_samples, sub2classes = iter(os.path.join(args.data_root, 'Process_gray_full_840_2class', 'FV_process_gray_2class'), training_samples, validating_samples, testing_samples, sub2classes)
	pickle.dump({
		'Training_Set':training_samples, 
		'Validating_Set':validating_samples, 
		'Testing_Set':testing_samples,
	}, open(args.annot_file, 'wb'))

	print(f'train_samples: {len(training_samples)}')
	print(f'val_samples: {len(validating_samples)}')
	print(f'test_samples: {len(testing_samples)}')
	print(training_samples[0])


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
	
