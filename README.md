
# Sistem Deteksi Vishing Otomatis dengan Whisper dan BERT

Proyek ini merupakan sistem deteksi voice phishing (vishing) otomatis berbasis transkripsi suara. Sistem ini menggunakan model *Faster-Whisper* diintegrasikan dengan whisper_streaming dan model *BERT* (melalui `VishingDetector`) untuk mengklasifikasikan apakah isi transkrip termasuk vishing atau bukan.

## ✨ Fitur Utama

- Transkripsi real-time dari mikrofon menggunakan `whisper_streaming`.
- Transkripsi dari file audio (offline).
- Klasifikasi kalimat hasil transkripsi untuk mendeteksi vishing.
- Perhitungan probabilitas dan peringatan jika kemungkinan vishing tinggi.

## ⚙️ Teknologi yang Digunakan

- Python 3
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) untuk transkripsi audio.
- `speech_recognition` dan `sounddevice` untuk audio real-time.
- PyTorch untuk inference model klasifikasi vishing.
- `VishingDetector` sebagai modul klasifikasi berbasis BERT yang telah dikembangkan.

## 🗂️ Struktur Folder (Singkat)

```
final_system/
│
├── main.py # Entry point sistem
├── final_model/
│ └── vishing_detector.py # Model klasifikasi vishing
│ └── model/ # Folder tempat model klasifikasi disimpan
├── whisper_streaming/
│ └── whisper_online.py # Modul ASR real-time
```

## 🚀 Cara Menjalankan

### 1. Clone Repository

```bash
git clone https://github.com/skripsi-kelar-sebulan/final_system.git
cd final_system
```

### 2. Install Dependencies

Gunakan `pip` dan pastikan environment Python Anda telah terinstal:

```bash
pip install -r requirements.txt
```

Karena model deteksi vishing berukuran besar dan tidak bisa dimasukkan langsung ke repository GitHub,  perlu diunduh secara manual dari Google Drive berikut:
[Link Google Drive Model Deteksi Vishing](https://drive.google.com/drive/folders/19mRauW9E5nKJPBQ7BA3zCq34qJl7qVuu?usp=drive_link)
Setelah diunduh, letakkan seluruh isi folder model tersebut ke dalam folder berikut pada project:
final_system/final_model/model/

### 3. Inisialisasi Submodule Git

Jika project menggunakan submodule Git (seperti `whisper_streaming`), jalankan perintah berikut setelah clone dan install dependencies:

```bash
git submodule update --init --recursive
```

**Penjelasan:**  
Perintah ini akan mengunduh dan menginisialisasi repository tambahan (submodule) yang digunakan di dalam project ini. Submodule adalah repository Git yang dimasukkan ke dalam repository utama sebagai dependensi kode. Dengan menjalankan perintah ini, kamu memastikan semua kode pendukung yang berada di submodule ikut ter-download dan siap digunakan.

### 4. Menjalankan Transkripsi dari File Audio

```bash
python main.py --audio_file path/to/audio.wav --model_type faster_whisper --model_size medium
```

### 5. Menjalankan Transkripsi Real-Time

```bash
python main.py --real_time --model_type faster_whisper --model_size medium
```

## 📋 Contoh Output

```
Starting real-time transcription and inference...
Transcription: Halo selamat siang, saya dari bank ingin menawarkan...
ALERT: High probability of Vishing detected!
```

## 👨‍💻 Kontributor

- Hanif, Ikra, Michael

## 📄 Lisensi

Proyek ini dirilis dengan lisensi [MIT License](LICENSE).  
Kamu bebas menggunakan, mengubah, dan menyebarluaskan proyek ini, selama tetap menyertakan lisensi asli.

## 🧾 Kredit

Sebagian komponen dalam proyek ini berasal dari [whisper_streaming](https://github.com/ufal/whisper_streaming) yang dilisensikan di bawah [MIT License](https://github.com/ufal/whisper_streaming/blob/main/LICENSE).
