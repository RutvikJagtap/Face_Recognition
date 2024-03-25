from django.urls import path
from . import views

urlpatterns = [
    path('members/', views.index, name='index'),
    path("process_video_frame/",views.process_video_frame,name="process_video_frame"),
]