from rest_framework import serializers

from food_watch.models import PictureEvent


class PictureEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = PictureEvent
        fields = '__all__'
