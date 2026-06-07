# Çoklu CNN Modelleri ile Fashion-MNIST Görüntü Sınıflandırması

Bu proje, Fashion-MNIST veri setindeki kıyafet görüntülerini sınıflandırmak için farklı CNN mimarilerinden çıkarılan özelliklerin birleştirilmesini amaçlar. Çalışmada tek CNN modeli kullanımı ile çoklu CNN tabanlı özellik füzyonu karşılaştırılmıştır.

## Projenin Amacı

Bu projede amaç, tek bir CNN modelinden elde edilen özellikler ile birden fazla CNN modelinden elde edilen özelliklerin birleştirilmesi durumunda sınıflandırma başarısının nasıl değiştiğini incelemektir.

Bu nedenle üç farklı CNN modeli kullanılmıştır:

- ResNet50
- MobileNetV2
- EfficientNetB0

Modeller doğrudan son sınıflandırma katmanlarıyla kullanılmamıştır. Bunun yerine her modelden özellik vektörleri çıkarılmış ve bu özellikler MLP sınıflandırıcı ile değerlendirilmiştir.

## Kullanılan Veri Seti

Projede Fashion-MNIST veri seti kullanılmıştır. Bu veri seti 10 farklı kıyafet sınıfından oluşur.

Sınıflar:

- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

Bilgisayar kaynaklarını çok zorlamamak için veri setinin tamamı yerine belirli sayıda örnek kullanılmıştır:

- Eğitim verisi: 4000 görüntü
- Doğrulama verisi: 800 görüntü
- Test verisi: 1200 görüntü

## Kullanılan Yöntemler

Projede dört temel işlem yapılmıştır:

1. **Feature Extraction**
   - CNN modellerinin son sınıflandırma katmanları kaldırılmıştır.
   - Modeller sadece görüntülerden özellik vektörü çıkarmak için kullanılmıştır.
   - `include_top=False` ve `pooling="avg"` kullanılmıştır.

2. **Feature Scaling**
   - Elde edilen özellik vektörleri `StandardScaler` ile ölçeklendirilmiştir.
   - Bunun amacı MLP sınıflandırıcının daha dengeli öğrenmesini sağlamaktır.

3. **Feature Fusion**
   - İki farklı özellik birleştirme yöntemi denenmiştir:
     - Concatenation
     - Weighted fusion

4. **Classification**
   - Çıkarılan tekli veya birleştirilmiş özellikler MLP sınıflandırıcıya verilmiştir.
   - MLP yapısında Dense ve Dropout katmanları kullanılmıştır.

## Fusion Yöntemleri

### Concatenation

Concatenation yönteminde farklı CNN modellerinden gelen özellik vektörleri yan yana eklenmiştir. Bu yöntem daha uzun ama daha fazla bilgi içeren bir özellik vektörü oluşturur.

### Weighted Fusion

Weighted fusion yönteminde özellik vektörleri ortak en küçük boyuta indirildikten sonra ağırlıklı olarak toplanmıştır. İki model birleşiminde ağırlıklar 0.5 ve 0.5, üç model birleşiminde ise 0.4, 0.3 ve 0.3 olarak kullanılmıştır.

## Transfer Learning ve Fine-tuning

Projede feature extraction dışında fine-tuning deneyleri de yapılmıştır. Fine-tuning aşamasında modelin tamamı eğitime açılmamış, sadece son katmanlardan bir kısmı güncellenmiştir.

Bu yöntemle hazır CNN modellerinin Fashion-MNIST veri setine daha fazla uyum sağlayıp sağlamadığı incelenmiştir.

## Kurulum

Gerekli kütüphaneleri yüklemek için aşağıdaki komut kullanılabilir:

```bash
pip install tensorflow keras numpy scikit-learn
```

## Çalıştırma

Python dosyasını çalıştırmak için:

```bash
python fashionmnist.py
```

Kod çalıştırıldığında:

- Fashion-MNIST veri seti yüklenir.
- ResNet50, MobileNetV2 ve EfficientNetB0 modellerinden özellik çıkarılır.
- Tek model, ikili model birleşimleri ve üçlü model birleşimleri denenir.
- Fine-tuning deneyleri çalıştırılır.
- Sonuçlar accuracy, precision, recall ve F1-score değerleriyle ekrana yazdırılır.

## Deney Sonuçları

En iyi genel sonuç, üç CNN modelinin concatenation yöntemiyle birleştirildiği deneyde elde edilmiştir.

| Deney | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Concat_3CNN_ResNet50+MobileNetV2+EfficientNetB0 | 0.8650 | 0.8697 | 0.8689 | 0.8685 |
| Weighted_2CNN_MobileNetV2+EfficientNetB0 | 0.8600 | 0.8674 | 0.8649 | 0.8651 |
| Single_MobileNetV2 | 0.8333 | 0.8426 | 0.8401 | 0.8371 |
| Single_EfficientNetB0 | 0.8300 | 0.8424 | 0.8364 | 0.8360 |
| Single_ResNet50 | 0.8192 | 0.8225 | 0.8254 | 0.8211 |

## Genel Değerlendirme

Deney sonuçlarına göre çoklu CNN özelliklerinin birleştirilmesi, tek CNN kullanımına göre daha başarılı sonuçlar verebilmiştir. Özellikle üç CNN modelinden gelen özelliklerin concatenation yöntemiyle birleştirilmesi en yüksek F1-score değerini vermiştir.

Fine-tuning sonuçları ise feature extraction ve fusion deneylerinin gerisinde kalmıştır. Bunun nedeni veri sayısının sınırlı olması, epoch sayısının düşük tutulması ve deneylerin CPU üzerinde çalıştırılması olabilir.

## Proje Dosyaları

Örnek dosya yapısı:

```text
│
├── CNN_FashionMnist.py
└── README.md
```

## Kullanılan Teknolojiler

- Python
- TensorFlow
- Keras
- NumPy
- scikit-learn
- Fashion-MNIST
- ResNet50
- MobileNetV2
- EfficientNetB0


