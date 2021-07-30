from django.shortcuts import render
import numpy as np
from PIL import Image
import joblib
from rest_framework.decorators import api_view
from django.http.response import JsonResponse
from rest_framework import status

# Load the previously trained model from the file
model = joblib.load("pretrained_model/mnist_model.pkl")

@api_view(['GET', 'POST', 'DELETE'])
def predict(request):  
    if request.method == 'POST':
        # Read the image uploaded by the curl command
        requested_img = request.FILES['file']

        '''
        Convert the uploaded image to greyscale.
        Since in MNIST the training images are greyscaled hence we will have to convert the uploaded image to greyscale
        '''
        greyscale_img = Image.open(requested_img).convert('L')

        '''
        Resize the uploaded image to 28x28 pixels.
        Since in MNIST the training images are of 28x28 pixels hence we will have to resize the uploaded image to 28x28 pixels.
        '''
        resized_image = greyscale_img.resize((28,28))

        # Convert the image to an array
        img = np.asarray(resized_image)

        # Reshape the image to (784, 1)
        img = img.reshape(784,)

        # Predict the digit using the trained model
        pred = model.predict(img.reshape(1, -1))

        # Get the digit
        result = int(pred.tolist()[0])

        
    # Return the JSON response
    return JsonResponse({"digit": result}, status=status.HTTP_201_CREATED)