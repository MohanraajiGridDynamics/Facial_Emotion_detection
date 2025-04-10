from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_run, name='home'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('face_count/', views.face_count_view, name='face_count'),
    path('direction/', views.direction_view, name='direction'),
    path('get_directions/', views.get_directions, name='get_directions'),
    path('get_emotion/', views.emotion_view, name='get_emotion'),

]
