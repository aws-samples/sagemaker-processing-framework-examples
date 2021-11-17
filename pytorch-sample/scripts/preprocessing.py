import shutil
import json
import pickle
import numpy as np
import torch
import torchvision as tv
import logging
import sys
from tqdm import tqdm
from pathlib import Path
from joblib import Parallel, delayed, parallel_backend
import script_utils

processing_dir = Path("/opt/ml/processing")

try:
    print(f'Script Name is {sys.argv[0]}')
    log_level = logging.INFO
    if len(sys.argv) >= 2:
        argv = sys.argv[1:]
        print(f'Arguments passed: {argv}')
        if argv[0].lower() == 'debug':
            log_level = logging.DEBUG
    
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%d-%m-%Y %H:%M:%S',
                        level=log_level,
                        filename='logs.txt')
    logger = logging.getLogger('sm-processing')

    logger.info('Starting pre-processing')

    # Set tmp folder to store intermediate results
    tmp_dir = processing_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Read input data from local folder
    input_data_dir = processing_dir / "input"
    shutil.unpack_archive((input_data_dir / "coco-annotations.zip").as_posix(), tmp_dir)
    print("Successfully downloaded COCO dataset annotations")
    logger.debug('Successfully downloaded COCO dataset annotations')

    # Load the annotations - training and validation datasets come in separate files
    with open((tmp_dir / "annotations/instances_train2017.json").as_posix(), "r") as f:
        train_metadata = json.load(f)

    with open((tmp_dir / "annotations/instances_val2017.json").as_posix(), "r") as f:
        val_metadata = json.load(f)

    # focus only on the images related to animals
    category_labels = {
        c["id"]: c["name"] for c in train_metadata["categories"] if c["supercategory"] == "animal"
    }
    print("Successfully loaded training and validation metadata")
    logger.debug('Successfully loaded training and validation metadata')

    # Extract metadata and image filepaths
    # For the train and validation sets, the data we need for the image labels and the filepaths are under different headings in the annotations. 
    # We have to extract each out and combine them into a single annotation in subsequent steps.
    train_annos = {}
    for a in train_metadata["annotations"]:
        if a["category_id"] in category_labels:
            train_annos[a["image_id"]] = {"category_id": a["category_id"]}

    train_images = {}
    for i in train_metadata["images"]:
        train_images[i["id"]] = {"coco_url": i["coco_url"], "file_name": i["file_name"]}

    val_annos = {}
    for a in val_metadata["annotations"]:
        if a["category_id"] in category_labels:
            val_annos[a["image_id"]] = {"category_id": a["category_id"]}

    val_images = {}
    for i in val_metadata["images"]:
        val_images[i["id"]] = {"coco_url": i["coco_url"], "file_name": i["file_name"]}


    # Combine label and filepath info
    # Later we’ll make our own train, validation and test splits. For this reason we’ll combine the training and validation datasets together.
    for id, anno in train_annos.items():
        anno.update(train_images[id])

    for id, anno in val_annos.items():
        anno.update(val_images[id])

    all_annos = {}
    for k, v in train_annos.items():
        all_annos.update({k: v})
    for k, v in val_annos.items():
        all_annos.update({k: v})

    # Sample the dataset
    # In order to make working with the data easier, we’ll select 250 images from each class at random. 
    # To make sure you get the same set of cell images for each run of this we’ll also set Numpy’s random seed to 0. 
    # This is a small fraction of the dataset, but it demonstrates how using transfer learning can give you good results without needing very large datasets.
    np.random.seed(0)
    sample_annos = {}

    for category_id in category_labels:
        subset = [k for k, v in all_annos.items() if v["category_id"] == category_id]
        sample = np.random.choice(subset, size=250, replace=False)
        for k in sample:
            sample_annos[k] = all_annos[k]

    # Download the images - 2500 images, takes ~ 5 minutes
    sample_dir = tmp_dir / "data_sample_2500"
    sample_dir.mkdir(exist_ok=True)
    with parallel_backend("threading", n_jobs=10):
        Parallel(verbose=3)(
            delayed(script_utils.download_image)(a["coco_url"], sample_dir) for a in sample_annos.values()
        )
    print("Successfully downloaded 2500 sample animal images from the COCO dataset")
    logger.debug('Successfully downloaded 2500 sample animal images from the COCO dataset')

    # Make train, validation and test splits 
    np.random.seed(0)
    image_ids = sorted(list(sample_annos.keys()))
    np.random.shuffle(image_ids)
    first_80 = int(len(image_ids) * 0.8)
    next_10 = int(len(image_ids) * 0.9)
    train_ids, val_ids, test_ids = np.split(image_ids, [first_80, next_10])

    struct_dir = tmp_dir / "data_structured"
    struct_dir.mkdir(exist_ok=True)

    # Create the train/val/test folder structure and copy image files 
    for name, split in zip(["train", "val", "test"], [train_ids, val_ids, test_ids]):
        split_dir = struct_dir / name
        split_dir.mkdir(exist_ok=True)
        for image_id in tqdm(split):
            category_dir = split_dir / f'{category_labels[sample_annos[image_id]["category_id"]]}'
            category_dir.mkdir(exist_ok=True)
            source_path = (sample_dir / sample_annos[image_id]["file_name"]).as_posix()
            target_path = (category_dir / sample_annos[image_id]["file_name"]).as_posix()
            shutil.copy(source_path, target_path)

    print("Successfully prepared the structure of the sample dataset")
    logger.debug('Successfully prepared the structure of the sample dataset')
    print("Full dataset size={}, train dataset size={}, validation dataset size={}, test dataset size={}".format(len(image_ids), len(train_ids), len(val_ids), len(test_ids)))
    logger.info('Full dataset size={}, train dataset size={}, validation dataset size={}, test dataset size={}'.format(len(image_ids), len(train_ids), len(val_ids), len(test_ids)))

    # Now prepare the images for training
    # define the resize and augmentation transformations
    data_transforms = {
        "train": tv.transforms.Compose(
            [
                tv.transforms.RandomResizedCrop(224),
                tv.transforms.RandomHorizontalFlip(p=0.5),
                tv.transforms.RandomVerticalFlip(p=0.5),
                tv.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                tv.transforms.ToTensor(),
            ]
        ),
        "val": tv.transforms.Compose(
            [tv.transforms.Resize(224), tv.transforms.CenterCrop(224), tv.transforms.ToTensor()]
        ),
    }

    # create the Pytorch dataset
    data_dir = struct_dir
    splits = ["train", "val"]

    datasets = {}
    for s in splits:
        datasets[s] = tv.datasets.ImageFolder(root=data_dir / s, transform=data_transforms[s])

    # create the Pytorch dataloaders for our datasets
    batch_size = 4
    shuffle = True
    num_workers = 4

    dataloaders = {}
    for s in splits:
        dataloaders[s] = torch.utils.data.DataLoader(
            datasets[s], batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
    
    # Resize all images now
    data_dir = struct_dir
    splits = ["train", "val", "test"]

    datasets = {}
    for s in splits:
        datasets[s] = tv.datasets.ImageFolder(
            root=data_dir / s,
            transform=tv.transforms.Compose([tv.transforms.Resize(224), tv.transforms.ToTensor()]),
        )
    
    print("All sample images successfully transformed")
    logger.debug('All sample images successfully transformed')

    # And save them to disk
    output_dir = processing_dir / "output"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    for s in splits:
        split_path = output_dir / s
        split_path.mkdir(exist_ok=True)
        for idx, (img_tensor, label) in enumerate(tqdm(datasets[s])):
            label_path = split_path / f"{label:02}"
            label_path.mkdir(exist_ok=True)
            filename = datasets[s].imgs[idx][0].split("/")[-1]
            tv.utils.save_image(img_tensor, label_path / filename)
    
    print("All images saved to disk")
    logger.debug('All images saved to disk')

except Exception as e:
    print(e)
    print("Issue preparing the dataset")
    logger.error(f'Issue preparing the dataset: {e}')
    pass

finally:
    log_dir = processing_dir / "logs"
    log_dir.mkdir(exist_ok=True, parents=True)
    logger.debug('Saving logfile to output dir')
    logger.info('Preprocessing done')
    shutil.copy('logs.txt', log_dir / 'logs.txt')

print("Finished running processing job")