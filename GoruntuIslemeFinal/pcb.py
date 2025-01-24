import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET

class PCBDevre_hata_tespit:
    def __init__(self, referans_yolu, test_yolu, xml_yolu):
        self.referans_yolu = referans_yolu
        self.test_yolu = test_yolu
        self.xml_yolu = xml_yolu
        self.referans_resmi, self.test_resmi = self.resimleri_yukle()
        self.test_resmi_adi = os.path.basename(test_yolu)
        
        # XML dosyasındaki etiketleri yükle
        self.kusur_etiketleri = self.xml_etiketlerini_yukle()


    def xml_etiketlerini_yukle(self):
        # XML dosyasını yükleyip etiketleri çıkar
        tree = ET.parse(self.xml_yolu)
        root = tree.getroot()
        kusur_etiketleri = []
        for object in root.findall('object'):
            name = object.find('name').text
            bndbox = object.find('bndbox')
            x_min = int(bndbox.find('xmin').text)
            y_min = int(bndbox.find('ymin').text)
            x_max = int(bndbox.find('xmax').text)
            y_max = int(bndbox.find('ymax').text)
            kusur_etiketleri.append((name, x_min, y_min, x_max, y_max))
        return kusur_etiketleri

    def resimleri_yukle(self):
        referans_resmi = cv2.imread(self.referans_yolu, cv2.IMREAD_GRAYSCALE)
        test_resmi = cv2.imread(self.test_yolu, cv2.IMREAD_GRAYSCALE)
        return referans_resmi, test_resmi

    def anahtar_noktalari_tespit_et_ve_eslestir(self):
        sift = cv2.SIFT_create()
        anahtar_noktalar1, tanimlayicilar1 = sift.detectAndCompute(self.referans_resmi, None)
        anahtar_noktalar2, tanimlayicilar2 = sift.detectAndCompute(self.test_resmi, None)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        eslesmeler = bf.match(tanimlayicilar1, tanimlayicilar2)
        eslesmeler = sorted(eslesmeler, key=lambda x: x.distance)
        return anahtar_noktalar1, anahtar_noktalar2, eslesmeler

    def resimleri_hizala(self):
        anahtar_noktalar1, anahtar_noktalar2, eslesmeler = self.anahtar_noktalari_tespit_et_ve_eslestir()
        iyi_eslesmeler = eslesmeler[:10]
        kaynak_noktalar = np.float32([anahtar_noktalar1[m.queryIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
        hedef_noktalar = np.float32([anahtar_noktalar2[m.trainIdx].pt for m in iyi_eslesmeler]).reshape(-1, 1, 2)
        M, maske = cv2.findHomography(hedef_noktalar, kaynak_noktalar, cv2.RANSAC, 5.0)
        yukseklik, genislik = self.referans_resmi.shape
        hizalanmis_resim = cv2.warpPerspective(self.test_resmi, M, (genislik, yukseklik))
        return hizalanmis_resim

    def kusurlari_tespit_et(self, hizalanmis_resim):
        fark = cv2.absdiff(self.referans_resmi, hizalanmis_resim)
        _, esiklenmis_fark = cv2.threshold(fark, 50, 255, cv2.THRESH_BINARY)
        konturlar, _ = cv2.findContours(esiklenmis_fark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        kusurlar = []
        for kontur in konturlar:
            x, y, w, h = cv2.boundingRect(kontur)
            kusurlar.append((x, y, w, h))
        return kusurlar

    def kusurlari_gorsellestir(self, hizalanmis_resim, kusurlar):
        olcek_faktoru = 0.5
        kucuk_referans_resmi = cv2.resize(self.referans_resmi, None, fx=olcek_faktoru, fy=olcek_faktoru, interpolation=cv2.INTER_AREA)
        kucuk_hizalanmis_resim = cv2.resize(hizalanmis_resim, None, fx=olcek_faktoru, fy=olcek_faktoru, interpolation=cv2.INTER_AREA)
        fark = cv2.absdiff(kucuk_referans_resmi, kucuk_hizalanmis_resim)
        
        kusur_resmi = cv2.cvtColor(kucuk_referans_resmi, cv2.COLOR_GRAY2BGR)
        # XML etiketlerine göre dikdörtgen çiz
        for (name, x_min, y_min, x_max, y_max) in self.kusur_etiketleri:
            cv2.rectangle(kusur_resmi, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        
        # Tespit edilen kusurları görselleştir
        for (x, y, w, h) in kusurlar:
            center_x = int(x * olcek_faktoru + w * olcek_faktoru / 2)
            center_y = int(y * olcek_faktoru + h * olcek_faktoru / 2)
            radius = max(int(w * olcek_faktoru / 2), int(h * olcek_faktoru / 2))  # Yarıçapı genişliğin veya yüksekliğin yarısı olarak al
            cv2.circle(kusur_resmi, (center_x, center_y), radius, (0, 0, 255), 3)  # Kırmızı yuvarlak çiz

        # Görselleri tek seferde göster
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.title('Referans Resim')
        plt.imshow(kucuk_referans_resmi)
        
        plt.subplot(2, 2, 2)
        plt.title('Hizalanmış Resim')
        plt.imshow(kucuk_hizalanmis_resim)
        
        plt.subplot(2, 2, 3)
        plt.title('Resimler Arasındaki Fark')
        plt.imshow(fark, cmap='plasma')
        
        plt.subplot(2, 2, 4)
        plt.title(f'Kusurlar - {self.test_resmi_adi}')
        plt.imshow(kusur_resmi)
        
        plt.show()  

detektor = PCBDevre_hata_tespit('C:\\Users\\zelih\\OneDrive\\Belgeler\\OPENCV\\GoruntuIslemeFinal\\PCB_DATASET\\Reference\\01.JPG', 'C:\\Users\\zelih\\OneDrive\\Belgeler\\OPENCV\\GoruntuIslemeFinal\\PCB_DATASET\\rotation\\Open_circuit_rotation\\01_open_circuit_13.jpg', 'C:\\Users\\zelih\\OneDrive\\Belgeler\\OPENCV\\GoruntuIslemeFinal\\PCB_DATASET\\annotated\\Open_circuit_rotation\\01_open_circuit_13.xml')
hizalanmis_resim = detektor.resimleri_hizala()
kusurlar = detektor.kusurlari_tespit_et(hizalanmis_resim)
detektor.kusurlari_gorsellestir(hizalanmis_resim, kusurlar)