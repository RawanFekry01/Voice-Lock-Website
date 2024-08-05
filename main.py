from flask import Flask
from flask import request
from flask import render_template,redirect, url_for
from functions import *


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('allow.html')


saragmm = pickle.load(open("Sara.gmm", "rb"))
rawangmm = pickle.load(open("rawan.gmm", "rb"))
salahgmm = pickle.load(open("salah.gmm", "rb"))

opengmm = pickle.load(open("openTheDoor.gmm", "rb"))
close = pickle.load(open("closeTheDoor.gmm", "rb"))
window = pickle.load(open("openTheWindow.gmm", "rb"))
please = pickle.load(open("pleaseOpen.gmm", "rb"))


@app.route("/preProcessing")
def preProcessing():

    features = extract_features('output.wav')

    sara = np.array(saragmm.score(features))
    rawan = np.array(rawangmm.score(features))
    salah = np.array(salahgmm.score(features))
    opendoorword = np.array(opengmm.score(features))
    closeword = np.array(close.score(features))
    windowword = np.array(window.score(features))
    pleaseword = np.array(please.score(features))

    wordscore = [opendoorword, closeword, windowword, pleaseword]
    wordresult = np.max(wordscore)

    speakerscore = [sara,rawan,salah]
    speakerResult=np.max(speakerscore)

    spectral_Rolloff('output.wav', 'spec_Rolloff')
    # spectral_Rolloff('GMM Data\salah\Mohamed_open_new (14).wav', 'Salah')
    print(speakerscore)
    plot_barChart(speakerscore, True, ['Sara', 'Rawan', 'Mohamed'],'Speaker')
    plot_barChart(wordscore, False, ['Open the door', 'Close the door', 'Open the window', 'Please open'], 'Word')    

    otherFlag = False
    otherscore = speakerscore - speakerResult

    for i in range(len(otherscore)):
        if otherscore[i] == 0 :
            continue
        if otherscore[i]>-1:
            otherFlag = True

    if otherFlag:
        speaker = 'Other' 
        predictedSpeaker = 'Nearest person' 
        flag = 'Rawan'
    else:
        if speakerResult == sara:
            speaker = 'Sara'
        elif speakerResult == rawan:
            speaker = "Rawan"
        else:
            speaker = "Mohamed"  
        predictedSpeaker = speaker
        flag = speaker
    if wordresult == opendoorword:
            wordIs = "Open"         
    else:
            wordIs = "other"

    
    prediction = speaker + " " + wordIs
    print(prediction)
    img,fig=plot_melspectrogram('output.wav')
    fig.colorbar(img,format="%+2.f")
    spectro= plt.savefig('./static/spectro.png')
    spectro=True

    return render_template('allow.html',prediction ="{}".format(prediction),spectro = spectro, predictedSpeaker = predictedSpeaker, flag = flag)


@app.route("/predict",methods = ['GET','POST'])
def predict():
    return redirect(url_for('preProcessing'))

    
@app.route("/",methods = ['GET','POST'])
def audio():
    if request.method == "POST":
        fs = 22050 # Sample rate
        seconds = 2 # Duration of recording
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
        sd.wait()  # Wait until recording is finished
        write('output.wav', fs, myrecording)
    return redirect(url_for('predict'))

if __name__ == "__main__":
    app.run(debug = True)


