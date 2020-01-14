from django.urls import path
from predictor import views
urlpatterns = [
    path('classify/', views.call_model.as_view())
]