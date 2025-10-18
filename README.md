# 🧠 Derin Öğrenme ile Görüntü İşleme Projeleri
* Bu repo,  bir dizi derin öğrenme projesini içermektedir.
* Proje seti, temel CNN uygulamalarından ileri seviye segmentasyon ve nesne tespiti modellerine kadar farklı seviyelerde görüntü işleme problemlerini kapsamaktadır.

## 🚀 Kullanılan Teknolojiler
* Python 3.10+
* TensorFlow / Keras
* OpenCV
* NumPy, Pandas, Matplotlib, Seaborn
* YOLOv8
* U-Net, CNN, Transfer Learning (VGG, ResNet)

## 📂 Proje İçerikleri
### 1️⃣ MNIST & Fashion MNIST Sınıflandırma
* El yazısı rakamlar (MNIST) ve giyim eşyaları (Fashion MNIST) veri setleri üzerinde CNN tabanlı sınıflandırma modelleri oluşturuldu.
* Eğitim, doğrulama ve test süreçleri optimize edildi.
* Model performansı doğruluk (accuracy) ve kayıp (loss) grafikleriyle değerlendirildi.

### 2️⃣ CNN Flowers
* Oxford Flowers 102 veri seti üzerinde çiçek türlerini sınıflandırmak için CNN mimarisi uygulandı.
* Data augmentation teknikleri ile veri çeşitliliği artırıldı.
* Modelin genel başarımı transfer learning ile geliştirildi.

### 3️⃣ Image Descriptions (Görsel Açıklama Üretimi)
* Bir görüntüye uygun açıklama üreten model geliştirildi.
* Görüntü özellikleri CNN (ör. VGG16) ile çıkarıldı.
* LSTM tabanlı metin üretimi modeli ile görsel açıklama oluşturuldu.
* Görüntü ve dil modellerinin birleşimiyle multimodal deep learning uygulaması yapıldı.

### 4️⃣ Bone Age Prediction
* Radyolojik el görüntülerinden kemik yaşını tahmin eden regresyon modeli geliştirildi.
* Transfer learning tabanlı CNN kullanıldı.
* Veri artırma, normalizasyon ve MAE metriğiyle değerlendirme yapıldı.
* Gerçek tıbbi veri senaryosuna uygun modelleme gerçekleştirildi.

### 5️⃣ Araç ve İnsan Tespiti (YOLOv8)
* Gerçek zamanlı nesne tespiti için YOLOv8 modeli kullanıldı.
* Araç, yaya ve benzeri nesnelerin tespiti için video/görsel veriler üzerinde test edildi.
* Çizgi geçiş sayımı ve hareket yönü tahmini gibi pratik senaryolar eklendi.

### 6️⃣ Görüntü Segmentasyonu (U-Net)
* Medikal görüntülerde piksel bazlı segmentasyon amacıyla U-Net mimarisi uygulandı.
* Mask veri setleri üzerinde model eğitildi.
* Görüntülerin detaylı biçimde ayrıştırılması sağlandı.
* Model performansı Dice ve IoU metrikleriyle değerlendirildi.
