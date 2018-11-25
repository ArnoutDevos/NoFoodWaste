from django.db import models
from jsonfield import JSONField


def picture_path(picture_patch, filename):
    return 'picture-events/{0}/{1}_{2}.jpg'.format(picture_patch.event.id, picture_patch.x, picture_patch.y)


class PictureEvent(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)


class PicturePatch(models.Model):
    x = models.IntegerField(blank=False, null=False)
    y = models.IntegerField(blank=False, null=False)

    picture = models.ImageField(upload_to=picture_path, blank=False, null=False)
    metadata = JSONField(default=dict)

    event = models.ForeignKey(PictureEvent, on_delete=models.CASCADE, related_name='patches')
