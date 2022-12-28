# Package untuk operasi web
import os
import secrets
from PIL import Image
from script import app
from flask import render_template, url_for, flash, redirect, request
from script.forms import ImageForm

# Package untuk ekstraksi ciri dari citra
import pickle
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2hsv
from skimage.morphology import binary_erosion, binary_dilation

# Fungsi untuk menyimpan gambar
def save_img(form_img):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(form_img.filename)
    img_fn = random_hex + f_ext
    img_path = os.path.join(app.root_path, 'static/pictures/uploads', img_fn)

    img = Image.open(form_img)
    img.save(img_path)

    return img_fn, img_path

# Fungsi untuk pemeriksaan key pada ekstraksi ciri
def check_key(dic):
    if len(dic.keys()) == 1:
        if list(dic.keys())[0] == True:
            dic[False] = 0
        else:
            dic[True] = 0
        return dic
    return dic

@app.route('/')
@app.route('/home')
def home():
    form = ImageForm()
    return render_template('home.html', form=form)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form = ImageForm()
        if form.validate_on_submit:
            # Menyimpan nama dan path gambar yang telah diunggah
            img_fn, img_path = save_img(form.image.data)

            # Mengambil gambar dari setiap folder
            img = imread(img_path)
            imgHSV = rgb2hsv(img)
            
            # Membuat mask untuk segmentasi

            # Mask seluruh buah
            lower_mask = imgHSV[:,:,0] <= 1
            upper_mask = imgHSV[:,:,0] >= 0
            saturation_mask = imgHSV[:,:,1] > 0.25
            all_mask = upper_mask * lower_mask * saturation_mask

            # Melakukan operasi morfologi pada mask buah
            morf_mask = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            all_mask = binary_dilation(all_mask, morf_mask)
            all_mask = binary_erosion(all_mask, morf_mask)

            # Mask matang
            # 1
            lower_mask = imgHSV[:,:,0] >= 0
            upper_mask = imgHSV[:,:,0] < 0.045
            saturation_mask = imgHSV[:,:,1] > 0.25
            value_mask = imgHSV[:,:,2] <= 1
            mature_mask_lower = lower_mask * upper_mask * saturation_mask * value_mask
            # 2
            lower_mask = imgHSV[:,:,0] > 0.9
            upper_mask = imgHSV[:,:,0] <= 1
            saturation_mask = imgHSV[:,:,1] > 0.25
            mature_mask_upper = lower_mask * upper_mask * saturation_mask
            # Final mask
            mature_mask = mature_mask_lower + mature_mask_upper

            # Mask belum matang
            lower_mask = imgHSV[:,:,0] > 0.175
            upper_mask = imgHSV[:,:,0] < 0.5
            saturation_mask = imgHSV[:,:,1] > 0.5
            immature_mask = lower_mask * upper_mask * saturation_mask

            # Mask busuk
            rotten_mask = all_mask ^ (mature_mask + immature_mask)

            # Ekstraksi ciri
            unique, counts = np.unique(all_mask, return_counts=True)
            allMaskCount = check_key(dict(zip(unique, counts)))[True]

            unique, counts = np.unique(mature_mask, return_counts=True)
            matureMaskCount = check_key(dict(zip(unique, counts)))[True]

            unique, counts = np.unique(rotten_mask, return_counts=True)
            rottenMaskCount = check_key(dict(zip(unique, counts)))[True]

            unique, counts = np.unique(immature_mask, return_counts=True)
            immatureMaskCount = check_key(dict(zip(unique, counts)))[True]

            # Menyimpan hasil ekstraksi pada array
            feature = pd.DataFrame({'mature': [matureMaskCount / allMaskCount], 
            'immature': [immatureMaskCount / allMaskCount], 
            'rotten': [rottenMaskCount / allMaskCount]})

            # Memanggil model klasifikasi KNN
            knn = pickle.load(open(os.path.join(app.root_path, 'static/cls.sav'), 'rb'))
            result = knn.predict(feature.to_numpy())

            # Mengembalikan hasil prediksi ke aplikasi
            if result == 0:
                flash('Citra apel terdeteksi matang!', 'success')
            elif result == 1:
                flash('Citra apel terdeteksi belum matang!', 'success')
            else:
                flash('Citra apel terdeteksi busuk!', 'success')
            
            img_file = url_for('static', filename=f'pictures/uploads/{img_fn}')

            return render_template('home.html', img_file=img_file, form=form)

    return redirect(url_for('home'))