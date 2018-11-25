import json
import shutil
import tempfile
from django.core.serializers import serialize
from io import BytesIO

import zipfile
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from watson_developer_cloud import VisualRecognitionV3

from food_watch.models import PictureEvent, PicturePatch
from food_watch.serializers import PictureEventSerializer

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='znv5zrsn6pxQsFGAOKtqMw6GJrIYz7XSblKEuN-Kih2d',
)


@api_view()
def hello_world(request):
    return Response({"message": "Hello, world!"})


def rank_classes(classes):
    return sorted(
        map(lambda x: {
            'class': x['class'],
            'score': x['score'],
        }, classes),
        key=lambda x: x['score'],
        reverse=True,
    )


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def watson_classification(patches, request):
    all_classes = {}

    for patches in batch(list(patches.keys()), 20):
        with tempfile.NamedTemporaryFile() as f:
            with BytesIO() as zip_file:
                with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for (x, y) in patches:
                        zipf.writestr('{}_{}'.format(x, y), request.data['patch_{}_{}'.format(x, y)].read())

                zip_file.seek(0)
                path = f.name
                shutil.copyfileobj(zip_file, f, length=131072)

            with open(path, 'rb') as zip_file:
                watson = visual_recognition.classify(
                    zip_file,
                    classifier_ids=["default"],
                ).get_result()

        for patch_data in watson['images']:
            classes = patch_data['classifiers'][0]['classes']
            x, y = map(int, patch_data['image'].split('/')[1].split('_'))
            all_classes[(x, y)] = rank_classes(classes)

    return all_classes


# def yolo_classification(request):
# from yolo.models import Darknet
# from yolo.utils.utils import load_classes
#     model = Darknet('../yolo/config/yolov3.cfg', img_size=416)
#     model.load_weights('../yolo/weights/yolov3.weights')
#
#     model.cuda()
#
#     model.eval()  # Set in evaluation mode
#
#     classes = load_classes('../yolo/data/coco.names')  # Extracts class labels from file
#
#     Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
#
#     imgs = []  # Stores image paths
#     img_detections = []  # Stores detections for each image index
#
#     print('\nPerforming object detection:')
#     prev_time = time.time()
#     for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
#         # Configure input
#         input_imgs = Variable(input_imgs.type(Tensor))
#
#         # Get detections
#         with torch.no_grad():
#             detections = model(input_imgs)
#             detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
#
#         # Log progress
#         current_time = time.time()
#         inference_time = datetime.timedelta(seconds=current_time - prev_time)
#         prev_time = current_time
#         print('\t+ Batch %d, Inference Time: %s' % (batch_i, inference_time))
#
#         # Save image and detections
#         imgs.extend(img_paths)
#         img_detections.extend(detections)
#
#     # Bounding-box colors
#     cmap = plt.get_cmap('tab20b')
#     colors = [cmap(i) for i in np.linspace(0, 1, 20)]
#
#     print('\nSaving images:')
#     # Iterate through images and save plot of detections
#     for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
#
#         print("(%d) Image: '%s'" % (img_i, path))
#
#         # Create plot
#         img = np.array(Image.open(path))
#         plt.figure()
#         fig, ax = plt.subplots(1)
#         ax.imshow(img)
#
#         # The amount of padding that was added
#         pad_x = max(img.shape[0] - img.shape[1], 0) * (opt.img_size / max(img.shape))
#         pad_y = max(img.shape[1] - img.shape[0], 0) * (opt.img_size / max(img.shape))
#         # Image height and width after padding is removed
#         unpad_h = opt.img_size - pad_y
#         unpad_w = opt.img_size - pad_x
#
#         # Draw bounding boxes and labels of detections
#         if detections is not None:
#             unique_labels = detections[:, -1].cpu().unique()
#             n_cls_preds = len(unique_labels)
#             bbox_colors = random.sample(colors, n_cls_preds)
#             for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
#                 print('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
#
#                 # Rescale coordinates to original dimensions
#                 box_h = ((y2 - y1) / unpad_h) * img.shape[0]
#                 box_w = ((x2 - x1) / unpad_w) * img.shape[1]
#                 y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
#                 x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
#
#                 color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
#                 # Create a Rectangle patch
#                 bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2,
#                                          edgecolor=color,
#                                          facecolor='none')
#                 # Add the bbox to the plot
#                 ax.add_patch(bbox)
#                 # Add label
#                 plt.text(x1, y1, s=classes[int(cls_pred)], color='white', verticalalignment='top',
#                          bbox={'color': color, 'pad': 0})
#
#         # Save generated image with detections
#         plt.axis('off')
#         plt.gca().xaxis.set_major_locator(NullLocator())
#         plt.gca().yaxis.set_major_locator(NullLocator())
#         plt.savefig('output/%d.png' % (img_i), bbox_inches='tight', pad_inches=0.0)
#         plt.close()


USE_WATSON = True


class PictureEventViewSet(viewsets.ModelViewSet):
    queryset = PictureEvent.objects.all()
    serializer_class = PictureEventSerializer

    def create(self, request, *args, **kwargs):
        patches = {}

        patches_data = json.loads(request.data['json']).pop('patches')
        event = PictureEvent.objects.create()
        for patch_data in patches_data:
            patch = PicturePatch(event=event, **patch_data)
            patch.picture = request.data['patch_{}_{}'.format(patch.x, patch.y)]
            patches[(patch.x, patch.y)] = patch

        all_classes = watson_classification(patches, request)

        for (x, y), detections in all_classes.items():
            patch = patches[(x, y)]
            patch.metadata = dict(detections=detections)
            patch.save()

        return Response(serialize('json', [event])[0], status=status.HTTP_201_CREATED)
