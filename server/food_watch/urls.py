from django.urls import path, include
from rest_framework import routers

from food_watch import views

router = routers.DefaultRouter(trailing_slash=False)
router.register('picture-events', views.PictureEventViewSet)

urlpatterns = [
    path('hello-world', views.hello_world),
    path('', include(router.urls)),
]