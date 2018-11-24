import json
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3(
    '2018-03-19',
    iam_apikey='znv5zrsn6pxQsFGAOKtqMw6GJrIYz7XSblKEuN-Kih2d')

with open('./image.jpg', 'rb') as images_file:
    classes = visual_recognition.classify(
        images_file,
        classifier_ids=["food"]).get_result()
    #print(json.dumps(classes, indent=2))
    with open('./response.json', 'w') as outfile:
        json.dump(classes, outfile)
