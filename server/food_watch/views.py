from rest_framework import viewsets
from rest_framework.decorators import api_view
from rest_framework.response import Response

from food_watch.models import PictureEvent
from food_watch.serializers import PictureEventSerializer


@api_view()
def hello_world(request):
    return Response({"message": "Hello, world!"})


class PictureEventViewSet(viewsets.ModelViewSet):
    queryset = PictureEvent.objects.all()
    serializer_class = PictureEventSerializer
