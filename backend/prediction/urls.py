from django.urls import path
from .views import predict, history, delete_prediction

urlpatterns = [
    path("predict/", predict),
    path("history/", history),

    # DELETE
    path("history/<int:pk>/", delete_prediction),
]