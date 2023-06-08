from flask import Flask,render_template,request
import rough
app=Flask(__name__)
@app.route('/')
def index():
    return(render_template('index.html'))
@app.route('/upload',methods=['POST'])
def upload():
    if request.method=='POST':
        image=request.files['image']
        result=rough.preprocess_image(image)
    return(render_template('result.html',result))

if __name__ =='__main__':
    app.run(debug=True)  
