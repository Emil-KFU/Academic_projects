from PyQt5 import QtWidgets, uic, QtCore, QtGui

import sys, os, time, functools, asyncio
import numpy as np
import cv2
import sqlite3
import PIL.Image as Img
import PIL.ImageEnhance as Enhance
from PIL import ImageDraw
import json

class ProcessingThread(QtCore.QThread):
    current_signal = QtCore.pyqtSignal(np.ndarray)
    cap = None
    pause = True

    def run(self):
        while True:
            if not self.pause:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()

                    if frame is None:
                        self.pause = True
                        continue

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    self.current_signal.emit(frame)

                else:
                    print('VideoCapture is None')

class UibdDialog(QtWidgets.QDialog):

    def push_button_add_click(self):
        material_name = self.lineEdit.text()
        material_area = self.lineEdit_2.text()
        material_area_std = self.lineEdit_3.text()
        material_porous = self.lineEdit_4.text()
        material_porous_std = self.lineEdit_5.text()

        data = [material_name, material_area, material_area_std, material_porous, material_porous_std]
        

        flag = [True if (m is not None and m != '') else False for m in data ]

        if flag:
            try:
                material_area = float(material_area)
                material_area_std = float(material_area_std)
                material_porous = float(material_porous)
                material_porous_std = float(material_porous_std)
                connect = sqlite3.connect(self.db_name)
                crsr = connect.cursor()
                crsr.execute("""INSERT INTO Materials(NAME,
                PORE_AREA_MEAN, PORE_AREA_STD, POROUS_MEAN, POROUS_STD)
                VALUES (?,?,?,?,?)""", (material_name, material_area, material_area_std, material_porous, material_porous_std))
                connect.commit()
                connect.close()
                self.load_materials()
                self.fill_table()

            except Exception as e:
                print(e)


    def push_button_delete_click(self):
        index = self.lineEdit_6.text()
        try:
            index = int(index) - 1
            if 0 <= index <= len(self.materials)-1:
                connect = sqlite3.connect(self.db_name)
                crsr = connect.cursor()
                row = self.materials.pop(index)
                id = row[0]
                crsr.execute('DELETE FROM Materials WHERE ID=?',(id,))
                connect.commit()
                connect.close()
                self.load_materials()
                self.fill_table()
        except Exception as e:
            print(e)

        self.lineEdit_6.setText('')
        self.lineEdit_6.clear()

    def load_materials(self):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        self.materials = cur.execute('SELECT * FROM Materials').fetchall()
        #print("materials =", self.materials)
        conn.close()

    def fill_table(self):
        while self.tableWidget.rowCount() > 0:
            self.tableWidget.removeRow(0)

        self.tableWidget.setColumnCount(6)
        self.tableWidget.setRowCount(len(self.materials))

        self.tableWidget.setHorizontalHeaderLabels(['ID', 'Наименование', 'Площадь поры', 'Откл. от площади',
                                                    'Пористость', 'Откл. от пористости'])

        self.tableWidget.horizontalHeaderItem(0).setToolTip("ID записи в базе данных")
        self.tableWidget.horizontalHeaderItem(1).setToolTip("Наименование материала")
        self.tableWidget.horizontalHeaderItem(2).setToolTip("Нормальная площадь поры")
        self.tableWidget.horizontalHeaderItem(3).setToolTip("Отклонение от нормы площади поры")
        self.tableWidget.horizontalHeaderItem(4).setToolTip("Нормальная пористость")
        self.tableWidget.horizontalHeaderItem(5).setToolTip("Отклонение от нормы пористости")

        for i, row in enumerate(self.materials):
            self.tableWidget.setItem(i, 0, QtWidgets.QTableWidgetItem(str(row[0])))
            self.tableWidget.setItem(i, 1, QtWidgets.QTableWidgetItem(str(row[1])))
            self.tableWidget.setItem(i, 2, QtWidgets.QTableWidgetItem(str(row[2])))
            self.tableWidget.setItem(i, 3, QtWidgets.QTableWidgetItem(str(row[3])))
            self.tableWidget.setItem(i, 4, QtWidgets.QTableWidgetItem(str(row[4])))
            self.tableWidget.setItem(i, 5, QtWidgets.QTableWidgetItem(str(row[5])))

        # делаем ресайз колонок по содержимому
        self.tableWidget.resizeColumnsToContents()

    def __init__(self, parent=None, db_name='bdValeev.db'):
        super(UibdDialog, self).__init__(parent)
       
        self.db_name = db_name
        self.parent = parent

        
        uic.loadUi('uibd.ui', self) 
        self.load_materials()

        self.fill_table() 


        self.pushButton.clicked.connect(self.push_button_add_click) 
        self.pushButton_2.clicked.connect(self.push_button_delete_click)
        

class UispprWindow(QtWidgets.QMainWindow):
    def explore(self, image):
        """
        Входной аргумент:
        image - исследуемое изображение
        Выход:
        image - изображение с контурами пор
        area_c - отношение площади всех пор ко всей площади изображения (пористость)
        len(bad_conrours) - количество 'плохих' пор
        """
        image = np.copy(image)
        # дополнительная обработка шумов
        blured = cv2.GaussianBlur(image, (5, 5), 0)
        # конвертация BGR формата в формат HSV
        hsv = cv2.cvtColor(blured, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([120, 120, 120])
        # определяем маску для обнаружения контуров пор.
        # будут выделены поры в заданном диапозоне
        mask = cv2.inRange(hsv, lower_black, upper_black)
        # получаем массив конутров
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_NONE)
        good_contours = []; bad_contours = []
        area_c = 0
        # находим 'хорошие' и 'плохие' поры
        for contour in contours:
            # также подсчитываем общую площадь пор
            area_c += cv2.contourArea(contour)
            if self.mat_area - self.mat_area_std <= cv2.contourArea(contour) <= self.mat_area + self.mat_area_std:
               good_contours.append(contour)
            else:
                bad_contours.append(contour) # т.е если пора слишком маленькая или слишком большая, то она "плохая"
        area_c = area_c / (image.shape[0] * image.shape[1])
        # выделяем 'хорошие' поры зеленым цветом
        cv2.drawContours(image, good_contours, -1, (0, 255, 0), 3)
        # выделяем 'плохие' поры красным цветом
        cv2.drawContours(image, bad_contours, -1, (255, 0, 0), 3)
        return image, area_c, len(bad_contours)
    @QtCore.pyqtSlot(np.ndarray)
    def set_current_frame(self, image):
        ''' Converts a QImage into an opencv MAT format '''
        self.set_original_frame(image)

    def set_original_frame(self, image):
        self.origin_img = cv2.resize(image, dsize=(300, 300))
        image = QtGui.QImage(self.origin_img.data, self.origin_img.shape[1], self.origin_img.shape[0],
                           QtGui.QImage.Format_RGB888)
        self.original_frame.setPixmap(QtGui.QPixmap.fromImage(image))
        
    def set_transformed_frame(self, image): # изменяет контрастность, яркость, резкость изображения
        image = Img.fromarray(image)
        # Перевод картинки в чёрно-белую:
        draw = ImageDraw.Draw(image) #Создаем инструмент для рисования.
        width = image.size[0] #Определяем ширину.
        height = image.size[1] #Определяем высоту.
        pix = image.load() #Выгружаем значения пикселей.
        for i in range(width):
            for j in range(height):
                a = pix[i, j][0]
                b = pix[i, j][1]
                c = pix[i, j][2]
                S = a + b + c
                if (S > (((255) // 2) * 3)):
                    a, b, c = 255, 255, 255
                else:
                    a, b, c = 0, 0, 0
                draw.point((i, j), (a, b, c))
        image = Enhance.Contrast(image).enhance(self.contrast_slider.value() / 10)
        image = Enhance.Brightness(image).enhance(self.brightness_slider.value() / 10)
        image = Enhance.Sharpness(image).enhance(self.sharpness_slider.value() / 10)
        image = np.array(image)
        self.transform_img = image
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        self.transformed_frame.setPixmap(QtGui.QPixmap.fromImage(image))

    def set_result_frame(self, image):

        self.result_img = image
        image, pore, bad_pores = self.explore(image) 
        image = QtGui.QImage(image.data, image.shape[1], image.shape[0], QtGui.QImage.Format_RGB888)
        if self.mat_porous - self.mat_porous_std <= pore <= self.mat_porous + self.mat_porous_std:
            pore_rep = 'в норме'
            self.label_16.setStyleSheet("QLabel { color: green }")
        else:
            pore_rep = 'не в норме'
            self.label_16.setStyleSheet("QLabel { color: red }")
        self.label_14.setText(str(pore))
        self.label_16.setText(pore_rep)
        self.label_20.setText(str(bad_pores))
        self.label_20.setStyleSheet("QLabel { color: green}")
        
        self.result_frame.setPixmap(QtGui.QPixmap.fromImage(image))

    def contrast_changed(self):
        self.set_transformed_frame(self.origin_img)
        self.set_result_frame(self.transform_img)

    def brightness_changed(self):
        self.set_transformed_frame(self.origin_img)
        self.set_result_frame(self.transform_img)

    def sharpness_changed(self):
        self.set_transformed_frame(self.origin_img)
        self.set_result_frame(self.transform_img)

    def material_selected(self, index, set_res_fr=True): 
        self.update_data(index)
        self.label_5.setText(str(self.mat_area))
        self.label_7.setText(str(self.mat_area_std))
        self.label_9.setText(str(self.mat_porous))
        self.label_11.setText(str(self.mat_porous_std))
        if set_res_fr:
            self.set_result_frame(self.transform_img)

    def create_db_if_not_exist(self):
        if not os.path.isfile(self.db_name):
            rows = [
                (0, 'Материал2', 12.0, 5.0, 0.1, 0.01),
                (1, 'Материал3', 9.00, 8.0, 0.15, 0.01),
                (2, 'Материал4', 15.0, 8.0, 0.2, 0.5),
                (3, 'Материал5', 14.0, 7.0, 0.3, 0.7),
            ]
            conn = sqlite3.connect(self.db_name)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE Materials 
                            (ID	INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                            NAME TEXT,
                            PORE_AREA_MEAN REAL NOT NULL,
                            PORE_AREA_STD REAL NOT NULL,
                            POROUS_MEAN REAL NOT NULL,
                            POROUS_STD REAL NOT NULL
                            )""")
            cur.executemany("""INSERT INTO Materials values (?,?,?,?,?,?)""", rows)
            conn.commit()
            conn.close()
        
    def load_materials(self):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        self.materials = cur.execute('SELECT * FROM Materials').fetchall()
        conn.close()

    def update_data(self, id: int):
        if len(self.materials) != 0: # self.materials - массив кортежей(записей) таблицы материалов
            row = self.materials[np.min([np.max([0, id]), len(self.materials) - 1])]
            self.mat_name = row[1]
            self.mat_area = row[2]
            self.mat_area_std = row[3]
            self.mat_porous = row[4]
            self.mat_porous_std = row[5]
        else:
            self.mat_name = 'Не задано'
            self.mat_area = 0
            self.mat_area_std = 0
            self.mat_porous = 0
            self.mat_porous_std = 0

    def UibdDialog_show(self):
       
        uibddialog = UibdDialog(self, self.db_name)
        uibddialog.show()
        
    def open_file(self):
        if self.thread:
            self.shoot_button.setEnabled(False)
            self.thread.pause = True
            if self.thread.cap is not None:
                self.thread.cap.release()
                self.thread.cap = None
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open file", QtCore.QDir.homePath())

        if filename != '':
            try:
                origin_img = Img.open(filename)
                origin_img = np.array(origin_img)

                self.set_original_frame(origin_img)
                self.set_transformed_frame(self.origin_img)
                self.set_result_frame(self.origin_img)
            except Exception as ex:
                print(ex)
            # PROCESS

    def set_via_webcam(self):
        if self.thread is None:
            self.thread = ProcessingThread(self)
            self.thread.current_signal.connect(self.set_current_frame)
            self.thread.start()
        if self.thread.cap is None:
            self.thread.cap = cv2.VideoCapture(0)
        if self.thread.pause is True:
            self.shoot_button.setEnabled(True)
            self.thread.pause = False

    
    def shoot_button_click(self):
        if self.thread.pause:
            self.thread.pause = False
        else:
            self.thread.pause = True
            self.set_transformed_frame(self.origin_img)
            self.set_result_frame(self.origin_img)
    
    def __init__(self):
        super(UispprWindow, self).__init__() # Call the inherited classes __init__ method
        self.db_name = 'bdValeev.db'
        self.create_db_if_not_exist() 
        uic.loadUi('uisppr.ui', self) 
        self.pushButton_3.clicked.connect(self.UibdDialog_show) 

        
        self.load_materials()
        for row in self.materials:
            self.comboBox.addItem(str(row[1]))
        self.comboBox.activated.connect(self.material_selected)


        self.actionWebcam_2.triggered.connect(self.set_via_webcam)
        self.actionWebcam_2.setStatusTip("Set webcam")

        self.actionOpen_2.triggered.connect(self.open_file)
        self.actionOpen_2.setStatusTip('Open file')


        self.original_frame = QtWidgets.QLabel(self.gridFrame) # в этот объект выводится изображение после вызова open_file
        self.original_frame.setMinimumSize(QtCore.QSize(300, 300))
        self.original_frame.setMaximumSize(QtCore.QSize(300, 300))
        self.original_frame.setObjectName("original_frame")


        self.transformed_frame = QtWidgets.QLabel(self.gridFrame_2)
        self.transformed_frame.setMinimumSize(QtCore.QSize(300, 300))
        self.transformed_frame.setMaximumSize(QtCore.QSize(300, 300))
        self.transformed_frame.setObjectName("transformed_frame")

        self.result_frame = QtWidgets.QLabel(self.gridFrame_3)
        self.result_frame.setMinimumSize(QtCore.QSize(300, 300))
        self.result_frame.setMaximumSize(QtCore.QSize(300, 300))
        self.result_frame.setObjectName("result_frame")


        self.shoot_button.clicked.connect(self.shoot_button_click)
        self.shoot_button.setEnabled(False)
        
        self.contrast_slider.setRange(-200, 200)
        self.contrast_slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.contrast_slider.setValue(10)

        self.brightness_slider.setRange(-50, 250)
        self.brightness_slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.brightness_slider.setValue(10)

        self.sharpness_slider.setRange(-200, 200)
        self.sharpness_slider.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.sharpness_slider.setValue(10)

        self.contrast_slider.valueChanged.connect(self.contrast_changed)
        self.brightness_slider.valueChanged.connect(self.brightness_changed)
        self.sharpness_slider.valueChanged.connect(self.sharpness_changed)


        self.material_selected(0, False)
        self.thread = None
        

class App(QtWidgets.QApplication):
    def __init__(self, *args):
        super(App, self).__init__(*args)
        self.main = UispprWindow()
        self.main.show()

if __name__ == "__main__":
    app = App(sys.argv)
    sys.exit(app.exec())

    
