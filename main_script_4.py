# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 10:30:18 2020

@author: arsen
"""

import sys
import gdal
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QPushButton,QWidget, QLineEdit,QFileDialog, QTextEdit, QTextBrowser, QPlainTextEdit                  
from PyQt5 import uic
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
from sklearn.cluster import KMeans
from matplotlib.backends.backend_qt5agg import FigureCanvas
import scipy as sy 
from scipy.signal import argrelextrema

uifile_0="../ui/main_window_4.ui"
uifile_1 = "../ui/GMM.ui" 
uifile_2 = "../ui/RGB.ui" 
uifile_3 = "../ui/Kmeans.ui" 
uifile_4 = "../ui/C_A_sen.ui"
uifile_5 = "../ui/C_A_lan.ui"

form_0, base_0 = uic.loadUiType(uifile_0)
form_1, base_1 = uic.loadUiType(uifile_1)
form_2, base_2 = uic.loadUiType(uifile_2)
form_3, base_3 = uic.loadUiType(uifile_3)
form_4, base_4 = uic.loadUiType(uifile_4)
form_5, base_5 = uic.loadUiType(uifile_5)


class Main(base_0, form_0):
    
    def __init__(self):
        super(base_0,self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openGMMWindow)
        self.pushButton_2.clicked.connect(self.openRGBWindow)
        self.pushButton_3.clicked.connect(self.openKmeansWindow)
        self.pushButton_4.clicked.connect(self.openC_A_sentinelWindow)
        self.pushButton_5.clicked.connect(self.openC_A_landsatWindow)
                    
    def openGMMWindow(self):
        self.GMMWindow=GMM_cluster()
        main.hide()
        self.GMMWindow.show()

    def openRGBWindow(self):
        self.RGBWindow=RGB()
        main.hide()
        self.RGBWindow.show()
        
    def openKmeansWindow(self):
        self.KmeansWindow=Kmeans_cluster()
        main.hide()
        self.KmeansWindow.show()
        
    def openC_A_sentinelWindow(self):
        self.C_A_sentinelWindow=C_A_sentinel()
        main.hide()
        self.C_A_sentinelWindow.show()
        
    def openC_A_landsatWindow(self):
        self.C_A_landsatWindow=C_A_landsat()
        main.hide()
        self.C_A_landsatWindow.show()
    
           
class GMM_cluster(base_1, form_1):
    
    def __init__(self):
        super(base_1,self).__init__()
        self.setupUi(self)                        
        self.pushButton.clicked.connect(self.file_open_dialogue)
        self.pushButton_2.clicked.connect(self.file_open_dialogue_save)
        self.pushButton_3.clicked.connect(self.return_fun)
        self.pushButton_4.clicked.connect(self.continue_fun)
        self.pushButton_5.clicked.connect(self.file_open_dialogue_save_2)
        
    def return_fun(self):
        GMM_cluster.hide(self)
        main.show()
    
    def continue_fun(self):
        self.build_plot_2()
        
    def file_open_dialogue(self):
        fileName_1 = QFileDialog.getOpenFileName(self,'Please, Band_2','','TIF files (*.tif)')
        fileName_2 = QFileDialog.getOpenFileName(self,'Please, Band_8','','TIF files (*.tif)')
        fileName_3 = QFileDialog.getOpenFileName(self,'Please, Band_12','','TIF files (*.tif)')       
        if fileName_1 [0]!='' and fileName_2 [0]!='':
            self.plainTextEdit.setPlainText(fileName_1[0])
            self.plainTextEdit.setPlainText(fileName_2[0])
            self.plainTextEdit.setPlainText(fileName_3[0])
            self.raster1=io.imread(fileName_1[0])
            self.raster2=io.imread(fileName_2[0])            
            self.raster3=io.imread(fileName_3[0])
            self.geodata= gdal.Open(fileName_3[0])            
            self.build_plot()
            self.build_plot_3()
            
    def DPC(self):
        SRTM_2=np.float64(resize(self.raster1, (self.raster3.shape[0], self.raster3.shape[1]), anti_aliasing=True))
        SRTM_8=np.float64(resize(self.raster2, (self.raster3.shape[0], self.raster3.shape[1]), anti_aliasing=True))
        SRTM_12 = np.float64(self.raster3)
        a = []
        a.append(SRTM_2)
        a.append(SRTM_8)
        a.append(SRTM_12)
        for i in a:
            col_mean = np.nanmean(i, axis = 0 )
            #Find indices that you need to replace
            inds_1 = np.where(np.isnan(i))
            inds_2 = np.where(i==0)
            #Place column means in the indices. Align the arrays using take
            i[inds_1] = np.take(col_mean, inds_1[1])
            i[inds_2] = np.take(col_mean, inds_2[1])                        
        quartz = SRTM_12/SRTM_2
        oil_spill = SRTM_12/SRTM_8
        quartz_2 = quartz.reshape(quartz.shape[0]*quartz.shape[1])
        oil_spill_2 = oil_spill.reshape(oil_spill.shape[0]*oil_spill.shape[1])
        df = np.column_stack((quartz_2, oil_spill_2))      
        X_std = StandardScaler().fit_transform(df)
        #initialize PCA with first 
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(X_std)        
        if ((pca.components_[1][0]) < 0 and (pca.components_[1][1]) > 0):
            print ("Signs are opposite")
            text = "Signs are opposite"           
        elif ((pca.components_[1][0]) > 0 and (pca.components_[1][1]) < 0):
            print ("Signs are opposite")
            text = "Signs are opposite"
        else:
            print ("Signs are not opposite")
            text = "Signs are not opposite"         
        self.label_3.setText(text)        
        [self.A,self.B] = np.hsplit(transformed, 2)
        self.B_2=self.B.reshape(oil_spill.shape[0], oil_spill.shape[1])            
        return self.B_2
           
    def Clustering(self):
        kmeans = KMeans(n_clusters=int(self.lineEdit.text()), init = 'k-means++', random_state=3).fit(self.DPC().reshape(-1,1))
        segmented = kmeans.labels_.reshape(self.B_2.shape[0], self.B_2.shape[1])
        self.spill = (np.bincount(segmented.reshape(self.B_2.shape[0] * self.B_2.shape[1])))*20**2/1000000    
        self.spill=[round(v,2) for v in self.spill]
        self.label_5.setText(str(self.spill))
        return segmented
        
    def build_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.scrollArea.setWidget(self.canvas)        
        self.figure.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.DPC(),cmap='jet')
        self.canvas.draw()
        
    def build_plot_2(self):
        self.figure.clear()
        self.ax.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        im = self.ax.imshow(self.Clustering(),cmap=plt.cm.get_cmap('tab10', int(self.lineEdit.text())))
        self.figure.colorbar(im, aspect=10, shrink=0.7)        
        self.canvas.draw()
    
    def build_plot_3(self):
        newdata=sy.ndimage.filters.gaussian_filter(self.B, (1.5,1.5))
        n_components = np.arange(1, 10)
        #Create an empty vector in which to store BIC scores
        wcss=np.zeros(n_components.shape)
        for i, n in enumerate(n_components):
            kmeans = KMeans(n_clusters= n, init = 'k-means++', random_state=3).fit(newdata)
            wcss[i] = kmeans.inertia_
            print('ok')
        self.figure_1 = plt.figure()
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.scrollArea_2.setWidget(self.canvas_1)        
        self.figure_1.clear()
        self.ax = self.figure_1.add_subplot(111)
        self.ax.set_xlim([1, 10])
        self.ax.set_title('The Elbow Method')
        self.ax.set_xlabel('Number of classes')
        self.ax.set_ylabel('Sum of sqr from points to center')
        self.ax.plot(n_components, wcss)
        self.canvas.draw()
        
    def file_open_dialogue_save(self):
        self.fileName_3 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_3 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_3[0])
        self.saveGeoTiff()   
            
    def saveGeoTiff(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_3[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.B_2)
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves
       
    def file_open_dialogue_save_2(self):
        self.fileName_5 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_5 [0]!='':
            self.plainTextEdit_2.clear()
            self.plainTextEdit_2.setPlainText(self.fileName_5[0])
        self.saveGeoTiff_2() 
            
    def saveGeoTiff_2(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_5[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.Clustering())
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves


class Kmeans_cluster(base_3, form_3):
    
    def __init__(self):
        super(base_3,self).__init__()
        self.setupUi(self)                       
        self.pushButton.clicked.connect(self.file_open_dialogue)
        self.pushButton_2.clicked.connect(self.file_open_dialogue_save)
        self.pushButton_3.clicked.connect(self.return_fun)
        self.pushButton_4.clicked.connect(self.continue_fun)
        self.pushButton_5.clicked.connect(self.file_open_dialogue_save_2)
               
    def return_fun(self):
        Kmeans_cluster.hide(self)
        main.show()
    
    def continue_fun(self):
        self.build_plot_2()
        
    def file_open_dialogue(self):
        fileName_1 = QFileDialog.getOpenFileName(self,'Please, Band_2','','TIF files (*.tif)')
        fileName_2 = QFileDialog.getOpenFileName(self,'Please, Band_5','','TIF files (*.tif)')
        fileName_3 = QFileDialog.getOpenFileName(self,'Please, Band_7','','TIF files (*.tif)')       
        if fileName_1 [0]!='' and fileName_2 [0]!='':
            self.plainTextEdit.setPlainText(fileName_1[0])
            self.plainTextEdit.setPlainText(fileName_2[0])
            self.plainTextEdit.setPlainText(fileName_3[0])
            self.raster1=io.imread(fileName_1[0])
            self.raster2=io.imread(fileName_2[0])            
            self.raster3=io.imread(fileName_3[0])
            self.geodata= gdal.Open(fileName_3[0])            
            self.build_plot()
            self.build_plot_3()
            
    def DPC(self):
        SRTM_2=np.float64(self.raster1)
        SRTM_8=np.float64(self.raster2)
        SRTM_12 = np.float64(self.raster3)        
        a = []
        a.append(SRTM_2)
        a.append(SRTM_8)
        a.append(SRTM_12)
        for i in a:
            col_mean = np.nanmean(i, axis = 0 )
            #Find indices that you need to replace
            inds_1 = np.where(np.isnan(i))
            inds_2 = np.where(i==0)
            #Place column means in the indices. Align the arrays using take
            i[inds_1] = np.take(col_mean, inds_1[1])
            i[inds_2] = np.take(col_mean, inds_2[1])                       
        quartz = SRTM_12/SRTM_2
        oil_spill = SRTM_12/SRTM_8
        quartz_2 = quartz.reshape(quartz.shape[0]*quartz.shape[1])
        oil_spill_2 = oil_spill.reshape(oil_spill.shape[0]*oil_spill.shape[1])
        df = np.column_stack((quartz_2, oil_spill_2))       
        X_std = StandardScaler().fit_transform(df)
        #initialize PCA with first 
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(X_std)
        if ((pca.components_[1][0]) < 0 and (pca.components_[1][1]) > 0):
            print ("Signs are opposite")
            text = "Signs are opposite"            
        elif ((pca.components_[1][0]) > 0 and (pca.components_[1][1]) < 0):
            print ("Signs are opposite")
            text = "Signs are opposite"
        else:
            print ("Signs are not opposite")
            text = "Signs are not opposite"         
        self.label_3.setText(text)
        [self.A,self.B] = np.hsplit(transformed, 2)
        self.B_2=self.B.reshape(oil_spill.shape[0], oil_spill.shape[1])            
        return self.B_2
            
    def Clustering(self):
        kmeans = KMeans(n_clusters=int(self.lineEdit.text()), init = 'k-means++', random_state=3).fit(self.DPC().reshape(-1,1))
        segmented = kmeans.labels_.reshape(self.B_2.shape[0], self.B_2.shape[1])
        self.spill = (np.bincount(segmented.reshape(self.B_2.shape[0] * self.B_2.shape[1])))*30**2/1000000    
        self.spill=[round(v,2) for v in self.spill]
        self.label_5.setText(str(self.spill))
        return segmented
        
    def build_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.scrollArea.setWidget(self.canvas)
        self.figure.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.DPC(),cmap='jet')
        self.canvas.draw()
        
    def build_plot_2(self):
        self.figure.clear()
        self.ax.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        im = self.ax.imshow(self.Clustering(),cmap=plt.cm.get_cmap('tab10', int(self.lineEdit.text())))
        self.figure.colorbar(im, aspect=10, shrink=0.7)
        self.canvas.draw()
    
    def build_plot_3(self):
        newdata=sy.ndimage.filters.gaussian_filter(self.B, (1.5,1.5))
        n_components = np.arange(1, 10)
        wcss=np.zeros(n_components.shape)
        for i, n in enumerate(n_components):
            kmeans = KMeans(n_clusters= n, init = 'k-means++', random_state=3).fit(newdata)
            wcss[i] = kmeans.inertia_
            print('ok')
        self.figure_1 = plt.figure()
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.scrollArea_2.setWidget(self.canvas_1)
        self.figure_1.clear()
        self.ax = self.figure_1.add_subplot(111)
        self.ax.set_xlim([1, 10])
        self.ax.set_title('The Elbow Method')
        self.ax.set_xlabel('Number of classes')
        self.ax.set_ylabel('Sum of sqr from points to center')
        self.ax.plot(n_components, wcss)
        self.canvas.draw()
        
    def file_open_dialogue_save(self):
        self.fileName_3 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_3 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_3[0])
        self.saveGeoTiff()   
           
    def saveGeoTiff(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_3[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.B_2)
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves
        
    def file_open_dialogue_save_2(self):
        self.fileName_3 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_3 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_3[0])
        self.saveGeoTiff_2()   
        
    def saveGeoTiff_2(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_3[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.Clustering())
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves

        
class RGB(base_2, form_2):
    
    def __init__(self):
        super(base_2,self).__init__()
        self.setupUi(self)       
        self.pushButton.clicked.connect(self.file_open_dialogue)
        self.pushButton_2.clicked.connect(self.file_open_dialogue_save)
        self.pushButton_3.clicked.connect(self.return_fun)
        
    def return_fun(self):
        RGB.hide(self)
        main.show()
    
    def file_open_dialogue(self):
        fileName_1 = QFileDialog.getOpenFileName(self,'Please, RED','','TIF files (*.tif)')
        fileName_2 = QFileDialog.getOpenFileName(self,'Please, GREEN','','TIF files (*.tif)')
        fileName_3 = QFileDialog.getOpenFileName(self,'Please, BLUE','','TIF files (*.tif)')
        if fileName_1 [0]!='' and fileName_2 [0]!='' and fileName_3 [0]!='':
            self.plainTextEdit.setPlainText(fileName_1[0])
            self.plainTextEdit.setPlainText(fileName_2[0])
            self.plainTextEdit.setPlainText(fileName_3[0])
            self.raster1=gdal.Open(fileName_1[0])
            self.raster = self.raster1.GetRasterBand(1).ReadAsArray()
            self.raster2=gdal.Open(fileName_2[0])
            self.raster_2 = self.raster2.GetRasterBand(1).ReadAsArray()
            self.raster3=gdal.Open(fileName_3[0])
            self.raster_3 = self.raster3.GetRasterBand(1).ReadAsArray()
            self.build_plot()
            
    def RGB_count(self):
        lminb=float(self.raster_3.min())
        lmaxb=float(self.raster_3.max())
        lming=float(self.raster_2.min())
        lmaxg=float(self.raster_2.max())
        lminr=float(self.raster.min())
        lmaxr=float(self.raster.max())
        norm_b=np.floor((self.raster_3-lminb)/(lmaxb-lminb)*255)
        norm_g=np.floor((self.raster_2-lming)/(lmaxg-lming)*255)
        norm_r=np.floor((self.raster-lminr)/(lmaxr-lminr)*255)
        self.rgb=np.dstack([norm_r,norm_g,norm_b]).astype(np.uint8)
        return self.rgb
        
    def build_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.scrollArea.setWidget(self.canvas)
        self.figure.clear()
        # create an axis
        ax = self.figure.add_subplot(111)
        ax.imshow(self.RGB_count())        
        self.canvas.draw() 
        
    def file_open_dialogue_save(self):
        self.fileName_4 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_4 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_4[0])
        self.saveGeoTiff()
    
    def saveGeoTiff(self):
        rows=self.rgb.shape[0]
        cols=self.rgb.shape[1]
        bands=self.rgb.shape[2]
        options = ['PHOTOMETRIC=RGB', 'PROFILE=GeoTIFF']
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_4[0], cols, rows, bands , gdal.GDT_UInt16,options=options)
        outdata.SetGeoTransform(self.raster1.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.raster1.GetProjection())##sets same projection as input
        for band in range(bands):
            outdata.GetRasterBand(band+1).WriteArray(self.rgb[:, :, band] )
            outdata.GetRasterBand(band+1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves         


class C_A_sentinel(base_4, form_4):
    
    def __init__(self):
        super(base_4,self).__init__()
        self.setupUi(self)                       
        self.pushButton.clicked.connect(self.file_open_dialogue)
        self.pushButton_2.clicked.connect(self.file_open_dialogue_save)
        self.pushButton_3.clicked.connect(self.return_fun)
        self.pushButton_4.clicked.connect(self.continue_fun)
        self.pushButton_5.clicked.connect(self.file_open_dialogue_save_2)
               
    def return_fun(self):
        C_A_sentinel.hide(self)
        main.show()
    
    def continue_fun(self):
        self.build_plot_2()
        
    def file_open_dialogue(self):
        fileName_1 = QFileDialog.getOpenFileName(self,'Please, Band_2','','TIF files (*.tif)')
        fileName_2 = QFileDialog.getOpenFileName(self,'Please, Band_8','','TIF files (*.tif)')
        fileName_3 = QFileDialog.getOpenFileName(self,'Please, Band_12','','TIF files (*.tif)')       
        if fileName_1 [0]!='' and fileName_2 [0]!='':
            self.plainTextEdit.setPlainText(fileName_1[0])
            self.plainTextEdit.setPlainText(fileName_2[0])
            self.plainTextEdit.setPlainText(fileName_3[0])
            self.raster1=io.imread(fileName_1[0])
            self.raster2=io.imread(fileName_2[0])            
            self.raster3=io.imread(fileName_3[0])
            self.geodata= gdal.Open(fileName_3[0])            
            self.build_plot()
            self.build_plot_3()
            
    def DPC(self):
        SRTM_2=np.float64(resize(self.raster1, (self.raster3.shape[0], self.raster3.shape[1]), anti_aliasing=True))
        SRTM_8=np.float64(resize(self.raster2, (self.raster3.shape[0], self.raster3.shape[1]), anti_aliasing=True))
        SRTM_12 = np.float64(self.raster3)        
        a = []
        a.append(SRTM_2)
        a.append(SRTM_8)
        a.append(SRTM_12)
        for i in a:
            col_mean = np.nanmean(i, axis = 0 )
            #Find indices that you need to replace
            inds_1 = np.where(np.isnan(i))
            inds_2 = np.where(i==0)
            #Place column means in the indices. Align the arrays using take
            i[inds_1] = np.take(col_mean, inds_1[1])
            i[inds_2] = np.take(col_mean, inds_2[1])                       
        quartz = SRTM_12/SRTM_2
        oil_spill = SRTM_12/SRTM_8
        quartz_2 = quartz.reshape(quartz.shape[0]*quartz.shape[1])
        oil_spill_2 = oil_spill.reshape(oil_spill.shape[0]*oil_spill.shape[1])
        df = np.column_stack((quartz_2, oil_spill_2))       
        X_std = StandardScaler().fit_transform(df)
        #initialize PCA with first 
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(X_std)
        if ((pca.components_[1][0]) < 0 and (pca.components_[1][1]) > 0):
            print ("Signs are opposite")
            text = "Signs are opposite"            
        elif ((pca.components_[1][0]) > 0 and (pca.components_[1][1]) < 0):
            print ("Signs are opposite")
            text = "Signs are opposite"
        else:
            print ("Signs are not opposite")
            text = "Signs are not opposite"         
        self.label_3.setText(text)
        [self.A,self.B] = np.hsplit(transformed, 2)
        self.B_1 = (np.array(self.B)-min(np.array(self.B)))/(max(np.array(self.B))-min(np.array(self.B)))
        self.B_2=self.B_1.reshape(oil_spill.shape[0], oil_spill.shape[1])            
        return self.B_2
            
    def Clustering(self):
        OX =[]
        OY =[]
        start = int(round(min(self.B_1)[0]))
        finish = int(round(max(self.B_1)[0]))
        for j in range (10*start, 10*finish,1):
            square = 0
            for i in self.B_1:
                if i >= j/10:
                    square +=1
            OY.append(square)
            OX.append(j/10)
            square = 0
        self.data = np.array(np.log10(OX[0:-1])-np.log10(OX[1:])/(np.log10(OY[0:-1])-np.log10(OY[1:])))
        self.maximum =[]
        for i in argrelextrema(self.data, np.greater)[0]:
            self.maximum.append(i)
        for i in argrelextrema(self.data, np.less)[0]:
            self.maximum.append(i)            
        self.border = []
        for i in self.maximum:
            #print(OX[i])
            self.border.append(OX[i])
        self.border.sort()
        klas = 0
        r,c=np.shape(self.B_2)
        self.claster=np.zeros((r,c),dtype=int)
        self.claster[self.B_2<self.border[0]]=klas
        self.claster[self.B_2>self.border[-1]]=len(self.border)
        for i in self.border:
            klas+=1
            self.claster[(self.B_2<(i+1))==(self.B_2>=i)]= klas
        
        return self.claster, self.data
        
    def build_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.scrollArea.setWidget(self.canvas)
        self.figure.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.DPC(),cmap='jet')
        self.canvas.draw()
        
    def build_plot_2(self):  
        self.figure.clear()
        self.ax.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        im = self.ax.imshow(self.Clustering()[0],cmap=plt.cm.get_cmap('tab10', len(self.border)+1))
        self.figure.colorbar(im, aspect=10, shrink=0.7)
        self.canvas.draw()
        self.spill = (np.bincount(self.Clustering()[0].reshape(self.B_2.shape[0] * self.B_2.shape[1])))*20**2/1000000
        self.spill=[round(v,2) for v in self.spill]
        self.label_5.setText(str(self.spill))
    
    def build_plot_3(self):        
        self.figure_1 = plt.figure()
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.scrollArea_2.setWidget(self.canvas_1)
        self.figure_1.clear()
        self.ax = self.figure_1.add_subplot(111)
        self.ax.set_title('Derivative')
        self.ax.set_ylabel('dA/dC')
        self.ax.plot(self.Clustering()[1])
        for i in range(len(self.maximum)):
            self.ax.scatter(self.maximum[i], self.Clustering()[1][self.maximum[i]], c='r')
        for i in self.maximum:
            self.ax.axvline(x=i, c='r')
        self.canvas.draw()
        
    def file_open_dialogue_save(self):
        self.fileName_4 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_4 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_4[0])
        self.saveGeoTiff()   
           
    def saveGeoTiff(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_4[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.B_2)
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves
        
    def file_open_dialogue_save_2(self):
        self.fileName_5 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_5 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_5[0])
        self.saveGeoTiff_2()   
        
    def saveGeoTiff_2(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_5[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.Clustering()[0])
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves


class C_A_landsat(base_5, form_5):
    
    def __init__(self):
        super(base_5,self).__init__()
        self.setupUi(self)                       
        self.pushButton.clicked.connect(self.file_open_dialogue)
        self.pushButton_2.clicked.connect(self.file_open_dialogue_save)
        self.pushButton_3.clicked.connect(self.return_fun)
        self.pushButton_4.clicked.connect(self.continue_fun)
        self.pushButton_5.clicked.connect(self.file_open_dialogue_save_2)
               
    def return_fun(self):
        C_A_landsat.hide(self)
        main.show()
    
    def continue_fun(self):
        self.build_plot_2()
        
    def file_open_dialogue(self):
        fileName_1 = QFileDialog.getOpenFileName(self,'Please, Band_2','','TIF files (*.tif)')
        fileName_2 = QFileDialog.getOpenFileName(self,'Please, Band_5','','TIF files (*.tif)')
        fileName_3 = QFileDialog.getOpenFileName(self,'Please, Band_7','','TIF files (*.tif)')       
        if fileName_1 [0]!='' and fileName_2 [0]!='':
            self.plainTextEdit.setPlainText(fileName_1[0])
            self.plainTextEdit.setPlainText(fileName_2[0])
            self.plainTextEdit.setPlainText(fileName_3[0])
            self.raster1=io.imread(fileName_1[0])
            self.raster2=io.imread(fileName_2[0])            
            self.raster3=io.imread(fileName_3[0])
            self.geodata= gdal.Open(fileName_3[0])            
            self.build_plot()
            self.build_plot_3()
            
    def DPC(self):
        SRTM_2=np.float64(self.raster1)
        SRTM_8=np.float64(self.raster2)
        SRTM_12 = np.float64(self.raster3)        
        a = []
        a.append(SRTM_2)
        a.append(SRTM_8)
        a.append(SRTM_12)
        for i in a:
            col_mean = np.nanmean(i, axis = 0 )
            #Find indices that you need to replace
            inds_1 = np.where(np.isnan(i))
            inds_2 = np.where(i==0)
            #Place column means in the indices. Align the arrays using take
            i[inds_1] = np.take(col_mean, inds_1[1])
            i[inds_2] = np.take(col_mean, inds_2[1])                       
        quartz = SRTM_12/SRTM_2
        oil_spill = SRTM_12/SRTM_8
        quartz_2 = quartz.reshape(quartz.shape[0]*quartz.shape[1])
        oil_spill_2 = oil_spill.reshape(oil_spill.shape[0]*oil_spill.shape[1])
        df = np.column_stack((quartz_2, oil_spill_2))       
        X_std = StandardScaler().fit_transform(df)
        #initialize PCA with first 
        pca = PCA(n_components=2)
        transformed = pca.fit_transform(X_std)
        if ((pca.components_[1][0]) < 0 and (pca.components_[1][1]) > 0):
            print ("Signs are opposite")
            text = "Signs are opposite"            
        elif ((pca.components_[1][0]) > 0 and (pca.components_[1][1]) < 0):
            print ("Signs are opposite")
            text = "Signs are opposite"
        else:
            print ("Signs are not opposite")
            text = "Signs are not opposite"         
        self.label_3.setText(text)
        [self.A,self.B] = np.hsplit(transformed, 2)
        self.B_2=self.B.reshape(oil_spill.shape[0], oil_spill.shape[1])            
        return self.B_2
            
    def Clustering(self):
        OX =[]
        OY =[]
        start = int(round(min(self.B)[0]))
        finish = int(round(max(self.B)[0]))
        for j in range (start, finish,1):
            square = 0
            for i in self.B:
                if i >= j:
                    square +=1
            OY.append(square)
            OX.append(j)
            square = 0
        self.data = np.array(np.log10(OX[0:-1])-np.log10(OX[1:])/(np.log10(OY[0:-1])-np.log10(OY[1:])))
        self.maximum =[]
        for i in argrelextrema(self.data, np.greater)[0]:
            self.maximum.append(i)
        for i in argrelextrema(self.data, np.less)[0]:
            self.maximum.append(i)            
        self.border = []
        for i in self.maximum:
            #print(OX[i])
            self.border.append(OX[i])
        self.border.sort()
        klas = 0
        r,c=np.shape(self.B_2)
        self.claster=np.zeros((r,c),dtype=int)
        self.claster[self.B_2<self.border[0]]=klas
        self.claster[self.B_2>self.border[-1]]=len(self.border)
        for i in self.border:
            klas+=1
            self.claster[(self.B_2<(i+1))==(self.B_2>=i)]= klas  
        return self.claster, self.data
        
    def build_plot(self):
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.scrollArea.setWidget(self.canvas)
        self.figure.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        self.ax.imshow(self.DPC(),cmap='jet')
        self.canvas.draw()
        
    def build_plot_2(self):
        self.figure.clear()
        self.ax.clear()
        # create an axis
        self.ax = self.figure.add_subplot(111)
        im = self.ax.imshow(self.Clustering()[0],cmap=plt.cm.get_cmap('tab10', len(self.border)+1))
        self.figure.colorbar(im, aspect=10, shrink=0.7)
        self.canvas.draw()
        self.spill = (np.bincount(self.Clustering()[0].reshape(self.B_2.shape[0] * self.B_2.shape[1])))*30**2/1000000
        self.spill=[round(v,2) for v in self.spill]
        self.label_5.setText(str(self.spill))
    
    def build_plot_3(self):        
        self.figure_1 = plt.figure()
        self.canvas_1 = FigureCanvas(self.figure_1)
        self.scrollArea_2.setWidget(self.canvas_1)
        self.figure_1.clear()
        self.ax = self.figure_1.add_subplot(111)
        self.ax.set_title('Derivative')
        self.ax.set_ylabel('dA/dC')
        self.ax.plot(self.Clustering()[1])
        for i in range(len(self.maximum)):
            self.ax.scatter(self.maximum[i], self.Clustering()[1][self.maximum[i]], c='r')
        for i in self.maximum:
            self.ax.axvline(x=i, c='r')
        self.canvas.draw()
        
    def file_open_dialogue_save(self):
        self.fileName_4 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_4 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_4[0])
        self.saveGeoTiff()   
           
    def saveGeoTiff(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_4[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.B_2)
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves
        
    def file_open_dialogue_save_2(self):
        self.fileName_5 = QFileDialog.getSaveFileName(self,'Please, direction','','TIF files (*.tif)')
        if self.fileName_5 [0]!='':
            self.plainTextEdit_2.setPlainText(self.fileName_5[0])
        self.saveGeoTiff_2()   
        
    def saveGeoTiff_2(self):
        rows,cols=np.shape(self.raster3)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(self.fileName_5[0], cols, rows, 1, gdal.GDT_Float64)
        outdata.SetGeoTransform(self.geodata.GetGeoTransform())##sets same geotransform as input
        outdata.SetProjection(self.geodata.GetProjection())##sets same projection as input
        outdata.GetRasterBand(1).WriteArray(self.Clustering()[0])
        outdata.GetRasterBand(1).SetNoDataValue(10000)##if you want these values transparent
        outdata.FlushCache() ##saves        
  

           
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())