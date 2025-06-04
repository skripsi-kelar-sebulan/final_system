from vishing_detector import VishingDetector
import time

if __name__ == '__main__':
    detector = VishingDetector()

    sample = "Selamat sore, saya dari Tim Keamanan Bank Nasional Indonesia. Kami menemukan aktivitas mencurigakan pada rekening Anda yang mengindikasikan adanya transaksi tidak sah sebesar 3 juta rupiah ke rekening luar negeri. Untuk mencegah pemblokiran sementara, mohon segera konfirmasi data Anda. Tolong sebutkan nama lengkap, nomor KTP, dan nomor rekening agar kami bisa proses verifikasi manual. Jika Anda tidak memberikan informasi ini dalam 10 menit, sistem keamanan otomatis kami akan mengunci rekening Anda untuk mencegah kerugian lebih lanjut. Kami mohon kerjasamanya demi keamanan bersama."
    start = time.time()
    result = detector.predict(sample)
    print(f"Inferensi selesai dalam {time.time() - start:.4f} detik")
    print(f'Text:\n{result["text"]}')
    print(f'Cleaned:\n{result["cleaned"]}')
    print(f'Prediction:\n{result["predicted_label"]}')
    print(f'Probability:\n{result["probability"]}')