from flask import Flask,render_template,request,send_from_directory
from werkzeug.utils import secure_filename
from ExpressionPredictor import PredictEmotion
import os


# UPLOAD_FOLDER = r'D:\Facial-CNN\Scripts\upload_files'

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path,'upload_files')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
UPLOAD_FOLDER = 'upload_files'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

#Second route : Use our model to make prediction - render the results page.
@app.route('/predict', methods=['Get','POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return render_template("error.html")
        file = request.files['file']

        # if user does not select file, browser also.
        # submit a empty part without filename
        if file.filename == '':
            return render_template("error.html")

        # Check whether the upoaded file is in allowed format.
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Upload the uploaded image into upload folder.
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            data = PredictEmotion(app,filename,filepath)
            return render_template(data['html'],image_name = data['image_name'], message= data['message'])
        else:
            return render_template("error.html")

   
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],filename)

@app.route('/refresh', methods=['POST'])
def refresh():
    return render_template("index.html")

                    
if __name__ == "__main__":
    app.run(debug=False,threaded=False)