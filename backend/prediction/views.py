from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import Prediction
from .serializers import PredictionSerializer


@api_view(["GET"])
def history(request):
    predictions = Prediction.objects.all().order_by("-id")
    serializer = PredictionSerializer(predictions, many=True)
    return Response(serializer.data)


@api_view(["POST"])
def predict(request):
    data = request.data

    predicted_yield = float(data["land_area"]) * 2.9

    Prediction.objects.create(
        crop=data["crop"],
        location=data["location"],
        land_area=data["land_area"],
        predicted_yield=predicted_yield,
    )

    return Response({
        "crop": data["crop"],
        "location": data["location"],
        "land_area": data["land_area"],
        "predicted_yield": predicted_yield,
    })


# NEW DELETE API
@api_view(["DELETE"])
def delete_prediction(request, pk):
    try:
        prediction = Prediction.objects.get(pk=pk)
    except Prediction.DoesNotExist:
        return Response(
            {"error": "Prediction not found"},
            status=status.HTTP_404_NOT_FOUND
        )

    prediction.delete()

    return Response(
        {"message": "Prediction deleted successfully"},
        status=status.HTTP_204_NO_CONTENT
    )