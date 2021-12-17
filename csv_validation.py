import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_annotations_path', help='Path to CSV annotations')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser.add_argument('--images_path',help='Path to images directory',type=str)
    parser.add_argument('--class_list_path',help='Path to classlist csv',type=str)
    parser.add_argument('--map',help='Use mAP instead of AP',type=str)
    parser.add_argument('--iou_threshold',help='IOU threshold used for evaluation',type=str, default='0.5')
    parser.add_argument('--save_path',help='Path where to save dictionaries',type=str)
    parser = parser.parse_args(args)

    #dataset_val = CocoDataset(parser.coco_path, set_name='val2017',transform=transforms.Compose([Normalizer(), Resizer()]))
    dataset_val = CSVDataset(parser.csv_annotations_path,parser.class_list_path,transform=transforms.Compose([Normalizer(), Resizer()]))
    # Create the model
    #if model was was saved wrapped in DataParellel the loaded model will already be wrapped in DataParallel too
    retinanet = torch.load(parser.model_path)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    if bool(parser.map):
        print(csv_eval.evaluate_mAP(dataset_val,retinanet,save_path=parser.save_path))
    else:
        print(csv_eval.evaluate(dataset_val, retinanet, iou_threshold=float(parser.iou_threshold)))



if __name__ == '__main__':
    main()
