from flask import Flask, render_template, request
import rough
import cv2
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return (render_template('index.html'))


@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        image = request.files['image']
        file_path = 'uploads/' + image.filename
        image.save(file_path)
        #print(file_path)
        result = rough.preprocess_image(image, target_size=(150, 150))
        return (render_template('result.html', result=result))


if __name__ == '__main__':
    app.run(debug=True)
