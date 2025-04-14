import pickle

import torch
import cv2
import os
import argparse
import tqdm
from sympy import false

import network.footandball as footandball
import data.augmentation as augmentations
from data.augmentation import PLAYER_LABEL, BALL_LABEL
from data.issia_utils import SequenceAnnotations, open_issia_sequence, read_issia_ground_truth, \
    evaluate_ball_detection_results, visualize_detection_results, save_detection_results
from misc.eval_config import Params


def get_boxes(detections):
    result = {}
    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        if label == PLAYER_LABEL:
            if "Person" not in result:
                result["Person"] = []
            result["Person"].append(box)

        elif label == BALL_LABEL:
            if "Ball" not in result:
                result["Ball"] = []
            result["Ball"].append(box)
    return result


def get_predictions(model: footandball.FootAndBall, params: Params):
    model.print_summary(show_architecture=False)
    model = model.to(params.device)

    _, file_name = os.path.split(params.path)

    if params.device == 'cpu':
        print('Loading CPU weights...')
        state_dict = torch.load(params.weights, map_location=lambda storage, loc: storage)
    else:
        print('Loading GPU weights...')
        state_dict = torch.load(params.weights)

    model.load_state_dict(state_dict)
    # Set model to evaluation mode
    model.eval()

    sequence = cv2.VideoCapture(params.path)
    n_frames = int(sequence.get(cv2.CAP_PROP_FRAME_COUNT))

    print('Processing video: {}'.format(params.path))
    pbar = tqdm.tqdm(total=n_frames)
    annotations = SequenceAnnotations()
    i = 0
    while sequence.isOpened():
        ret, frame = sequence.read()
        if not ret:
            # End of video
            break

        # Convert color space from BGR to RGB, convert to tensor and normalize
        img_tensor = augmentations.numpy2tensor(frame)

        with torch.no_grad():
            # Add dimension for the batch size
            img_tensor = img_tensor.unsqueeze(dim=0).to(params.device)
            detections = model(img_tensor)[0]

        boxes = get_boxes(detections)
        if 'Ball' in boxes:
            for box in boxes['Ball']:
                x1, y1, x2, y2 = box
                x = int((x1 + x2) / 2)
                y = int((y1 + y2) / 2)
                annotations.ball_pos[i].append((x, y))
        if 'Person' in boxes:
            for bpx in boxes['Person']:
                continue
        pbar.update(1)
        i += 1

    pbar.close()
    sequence.release()
    return annotations

if __name__ == '__main__':
    print('Run FootAndBall evaluation')

    # Train the DeepBall ball detector model
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='path to config file', type=str, required=True)
    # parser.add_argument('--profile', help='path to config file', type=str, default='ORIGINAL')
    parser.add_argument('--profile', help='profile form config', type=str, default='MODIFIED')
    parser.add_argument('--save_detections', help='save frames with predictions and gt', type=bool, default=False)
    # parser.add_argument('--path', help='path to video', type=str, required=True)
    # parser.add_argument('--model', help='model name', type=str, default='fb2')
    # parser.add_argument('--weights', help='path to model weights', type=str, required=True)
    # parser.add_argument('--ball_threshold', help='ball confidence detection threshold', type=float, default=0.7)
    # parser.add_argument('--player_threshold', help='player confidence detection threshold', type=float, default=0.7)
    # parser.add_argument('--device', help='device (CPU or CUDA)', type=str, default="cuda:0")
    # parser.add_argument('--annotations_file', help='file to save/load the annotations', type=str, default="se_cbam_5_annotations.pkl")
    args = parser.parse_args()
    params = Params(args.config, profile=args.profile)

    print('Video path: {}'.format(params.path))
    print('Model: {}'.format(params.model))
    print('Model weights path: {}'.format(params.weights))
    print('Ball confidence detection threshold [0..1]: {}'.format(params.ball_threshold))
    print('Player confidence detection threshold [0..1]: {}'.format(params.player_threshold))
    print('Device: {}'.format(params.device))
    print('')

    assert os.path.exists(params.weights), 'Cannot find FootAndBall model weights: {}'.format(params.weights)
    assert os.path.exists(params.path), 'Cannot open video: {}'.format(params.path)

    model = footandball.model_factory(params.model, 'detect', ball_threshold=params.ball_threshold,
                                      player_threshold=params.player_threshold)

    # File to save/load the SequenceAnnotations object

    # Check if the file already exists
    if os.path.exists(params.annotations_file):
        # Load the SequenceAnnotations object from the file
        with open(params.annotations_file, "rb") as f:
            annotations = pickle.load(f)
            print("Loaded annotations from file.")
    else:
        # Call `get_predictions` to compute the SequenceAnnotations
        annotations = get_predictions(model, params)

        # Save the SequenceAnnotations object to the file
        with open(params.annotations_file, "wb") as f:
            pickle.dump(annotations, f)
            print("Saved annotations to file.")
    # Example to demonstrate usage of module procedures
    print("OpenCV version: " + str(cv2.__version__))

    # Read ISSIA sequence and visualize ground truth
    # ISSIA dataset can be downloaded from http://www.issia.cnr.it/wp/dataset-cnr-fig/
    # Camera ids are between 1 and 6
    dataset_path = '/Users/sergebishyr/PhD/datasets/issia'
    camera_id = params.camera
    # sequence = open_issia_sequence(camera_id, dataset_path)

    # Read annotations included in the dataset

    gt_annotations = read_issia_ground_truth(camera_id, dataset_path)
    #
    # # Show annotated video sequence
    if args.save_detections:
        save_detection_results(camera_id, dataset_path, gt_annotations=gt_annotations, annotations=annotations)

    # Ball detection in pixels performance
    # This should return all ones as we evaluate the performance on ground truth data
    tolerance = 5
    avg_precision, avg_recall, percent_correctly_classified_frames = evaluate_ball_detection_results(annotations,
                                                                                                     gt_annotations,
                                                                                                     tolerance=tolerance)

    print('Avg. precision = ' + str(avg_precision))
    print('Avg. recall = ' + str(avg_recall))
    print('Percent of correctly classified frames = ' + str(percent_correctly_classified_frames))
