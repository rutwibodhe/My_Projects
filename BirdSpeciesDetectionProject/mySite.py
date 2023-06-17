# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request,session,Response
from werkzeug import secure_filename
from supportFile import predict  
from soundTest import soundPredict
import os
import cv2
import webbrowser

app = Flask(__name__)

app.secret_key = '1234'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/home', methods=['GET', 'POST'])
def home():
	return redirect(url_for('input'))

@app.route('/', methods=['GET', 'POST'])
def input():
	error = None
	if request.method == 'POST':
		if request.form['username'] != 'admin' or request.form['password'] != 'admin':
			error = 'Invalid Credentials. Please try again.'
		else:
			return redirect(url_for('image'))

	return render_template('input.html', error=error)



@app.route('/image', methods=['GET', 'POST'])
def image():
	global result
	if request.method == 'POST':
		if request.form['sub']=='Upload':
			savepath = r'upload/'
			photo = request.files['photo']
			photo.save(os.path.join(savepath,(secure_filename(photo.filename))))
			image = cv2.imread(os.path.join(savepath,secure_filename(photo.filename)))
			cv2.imwrite(os.path.join("static/images/","test_image.jpg"),image)
			return render_template('image.html')
		elif request.form['sub'] == 'Test':
			result = predict()
			return render_template('image.html',result=result)
		elif request.form['sub'] == 'Show Info':
			webbrowser.open_new_tab('https://google.com/search?q='+result+" bird")
	return render_template('image.html')

@app.route('/sound', methods=['GET', 'POST'])
def sound():
	global result
	if request.method == 'POST':
		if request.form['sub']=='Upload':
			savepath = r'upload/'
			dataset = request.files['dataset']
			dataset.save(os.path.join(savepath,(secure_filename('test.wav'))))
			return render_template('sound.html',mgs="File Uploaded..!!!")

		elif request.form['sub'] == 'Test':
			result = soundPredict()
			return render_template('sound.html',result=result)
		elif request.form['sub'] == 'Show Info':
			webbrowser.open_new_tab('https://google.com/search?q='+result+" bird")
	return render_template('sound.html')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
	# response.cache_control.no_store = True
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)
