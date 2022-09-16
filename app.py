from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np


app = Flask(__name__)           # created flask app
@app.route('/',methods=['GET'])      # routes to home(root directory)
def home():
    return render_template("web.html")


@app.route('/prediction',methods=['POST'])
def predict():
    img = request.files['dog_image']
    img_path = 'static/' + img.filename
    img.save(img_path)

    model = tf.keras.models.load_model('models/model1')  # specifically this code to load all custom objects

    image = tf.io.read_file(img_path)       # read image as a tensor
    image_tensor = tf.image.decode_jpeg(image, channels=3)      # decode to 3 channels (RGB)
    image = tf.image.convert_image_dtype(image_tensor, tf.float32)      # convert values to float32
    image = tf.image.resize(image, size=(224, 224))     # resize to (224, 224)
    image = tf.data.Dataset.from_tensors((tf.constant(image)))      # converts to a constant tensor
    batch_image = image.batch(1)        # turns the image data into a batch

    pred = model.predict(batch_image)       # gives an array of 120. Each reqresenting a unique breed

    unique_breeds = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 
    'american_staffordshire_terrier', 'appenzeller',
    'australian_terrier', 'basenji', 'basset', 'beagle',
    'bedlington_terrier', 'bernese_mountain_dog',
    'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
    'bluetick', 'border_collie', 'border_terrier', 'borzoi',
    'boston_bull', 'bouvier_des_flandres', 'boxer',
    'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
    'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
    'chow', 'clumber', 'cocker_spaniel', 'collie',
    'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
    'doberman', 'english_foxhound', 'english_setter',
    'english_springer', 'entlebucher', 'eskimo_dog',
    'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
    'german_short-haired_pointer', 'giant_schnauzer',
    'golden_retriever', 'gordon_setter', 'great_dane',
    'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
    'ibizan_hound', 'irish_setter', 'irish_terrier',
    'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
    'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
    'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
    'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
    'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
    'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
    'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
    'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
    'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
    'saint_bernard', 'saluki', 'samoyed', 'schipperke',
    'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
    'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
    'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
    'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
    'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
    'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
    'west_highland_white_terrier', 'whippet',
    'wire-haired_fox_terrier', 'yorkshire_terrier']

    pred_label = unique_breeds[np.argmax(pred)]     # gives the maximum predicted value from all the 120 elements

    return render_template("web.html", dog_image = img_path, prediction = pred_label)


if __name__ == '__main__':
    app.run(debug = True)