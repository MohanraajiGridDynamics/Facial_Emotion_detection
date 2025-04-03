from django.urls import path
from .views import home, detect_emotion

urlpatterns = [
    path('', home, name='home'),
    path('detect_emotion/', detect_emotion, name='detect_emotion'),
]
