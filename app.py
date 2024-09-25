from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

class PlateFinder:
    def __init__(self, minPlateArea, maxPlateArea):
        self.minPlateArea = minPlateArea
        self.maxPlateArea = maxPlateArea
        self.corresponding_area = []

    def find_possible_plates(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

        plate_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w)/h
            if area > self.minPlateArea and area < self.maxPlateArea and aspect_ratio > 2 and aspect_ratio < 5:
                plate_contours.append(contour)
                self.corresponding_area.append((x, y, w, h))

        return plate_contours

@app.route('/find_plates', methods=['POST'])
def find_plates():
    if request.method == 'POST':
        img = request.files['image']
        img_array = np.fromstring(img.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        plate_finder = PlateFinder(2000, 10000)  # adjust minPlateArea and maxPlateArea as needed
        possible_plates = plate_finder.find_possible_plates(img)

        response = []
        for contour in possible_plates:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_plate = img[y:y+h, x:x+w]
            _, img_encoded = cv2.imencode('.jpg', cropped_plate)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            response.append({'plate': img_base64})

        return jsonify({'plates': response})

if __name__ == '__main__':
    app.run(debug=True)
