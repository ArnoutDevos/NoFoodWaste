from rest_framework import serializers

from food_watch.models import PictureEvent, PicturePatch


class PicturePatchSerializer(serializers.ModelSerializer):
    class Meta:
        model = PicturePatch
        exclude = ('event',)


class PictureEventSerializer(serializers.ModelSerializer):
    patches = PicturePatchSerializer(many=True, read_only=False)

    class Meta:
        model = PictureEvent
        fields = '__all__'

    def create(self, validated_data):
        patches_data = validated_data.pop('patches')
        event = PictureEvent.objects.create(**validated_data)
        for patch_data in patches_data:
            PicturePatch.objects.create(event=event, **patch_data)
        return event
