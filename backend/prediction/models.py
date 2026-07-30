from django.db import models


class Prediction(models.Model):
    crop = models.CharField(max_length=100)
    location = models.CharField(max_length=100)
    land_area = models.FloatField()
    predicted_yield = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crop} - {self.location}"