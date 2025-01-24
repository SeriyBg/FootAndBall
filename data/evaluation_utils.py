import torch
from data.issia_utils import SequenceAnnotations


def build_annotations_from_predictions(predictions, delta=-8):
    annotations = SequenceAnnotations()

    # Process ball positions
    ball_confidence_map = predictions[0]
    for frame_idx in range(ball_confidence_map.size(0)):
        confidence = ball_confidence_map[frame_idx, 0]
        max_conf, max_idx = torch.max(confidence.view(-1), dim=0)
        y, x = divmod(max_idx.item(), confidence.size(1))
        annotations.ball_pos[frame_idx + delta].append((x, y))

    # Process ball shots
    ball_shot_map = predictions[1]
    threshold = 0.5
    for frame_idx in range(ball_shot_map.size(0)):
        shot_confidence = ball_shot_map[frame_idx, 0].mean().item()
        annotations.ball_shot[frame_idx + delta] = shot_confidence > threshold

    # Process interacting players
    interacting_players_map = predictions[2]
    for frame_idx in range(interacting_players_map.size(0)):
        player_ids = torch.where(interacting_players_map[frame_idx] > 0.5)[0]
        annotations.interacting_player[frame_idx + delta].extend(player_ids.tolist())

    # Process player bounding boxes
    person_bbox_map = predictions[2]
    for frame_idx in range(person_bbox_map.size(0)):
        for bbox_idx in range(person_bbox_map.size(1)):
            bbox = person_bbox_map[frame_idx, bbox_idx]
            height, width, x, y = bbox.tolist()
            annotations.persons[frame_idx + delta].append((bbox_idx, height, width, x, y))

    return annotations