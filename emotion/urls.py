from django.urls import path
from .views import home, detect_emotion, count_faces

urlpatterns = [
    path('', home, name='home'),
    path('detect_emotion/', detect_emotion, name='detect_emotion'),
    path('count_faces/', count_faces, name='count_faces'),

]
