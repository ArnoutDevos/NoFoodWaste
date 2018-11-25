from django.db import models
from jsonfield import JSONField
from uuid import uuid4


def picture_path(picture_event, filename):
    return 'picture-events/{0}.jpg'.format(str(uuid4()))


class PictureEvent(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    picture = models.ImageField(upload_to=picture_path, blank=False, null=False)
    metadata = JSONField(default=dict)
