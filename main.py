import argparse
import time
from whisper_streaming.whisper_online import asr_factory, OnlineASRProcessor

import math
import speech_recognition as sr
import numpy as np
import sounddevice as sd
from queue import Queue
import torch.nn.functional as F 
import threading
import psutil, os
import torch

def main(audio_file, model_type, model_size, debug, log_file, real_time, chunk_duration=2, ):
    """
    Main function for transcription and vishing detection.

    Args:
        audio_file (str): Path to the audio file (None for real-time transcription).
        model_type (str): Type of Whisper model (e.g., "faster_whisper").
        model_size (str): Size of the Whisper model (e.g., "large-v2").
        debug (bool): Enable debugging logs.
        log_file (str): Path to the log file.
        real_time (bool): Whether to run in real-time mode.
    """
    start_time = time.time()
    transcription_result = None  # Initialize transcription result

    if debug:
        log_performance("Start", log_file)

    if real_time:
        # Real-time transcription and inference using microphone
        

        print("Starting real-time transcription and inference...")

        
        print("Real-time transcription and vishing detection started. Press Ctrl+C to stop.")

        prediction = 0

        transcription_history = ""
        
        # Initialize the SpeechRecognizer
        recorder = sr.Recognizer()
        recorder.energy_threshold = 1000
        recorder.dynamic_energy_threshold = False

        # Set up the microphone
        mic = sr.Microphone(sample_rate=16000)

        audio_queue = Queue()
        
        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            audio_queue.put(data)
            

        source = sr.Microphone(sample_rate=16000)
        listener = recorder.listen_in_background(source, record_callback, phrase_time_limit=1)

        # Adjust for ambient noise
        with mic:
            recorder.adjust_for_ambient_noise(mic)

    
        threshold = 0.8
        counter = time.time()
        detected_flag = False
        stop_event = threading.Event()
        audio_start_wall_time = time.time()
        flag_stop = False

        try:
            while True:    
                if not audio_queue.empty():
                    audio_chunk = audio_queue.get()
                    print(f"Memory usage (MB): {psutil.Process(os.getpid()).memory_info().rss / 1024**2:.2f}")
                    print("Audio captured, processing...")

                    # Convert audio to raw PCM data
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

                    # Insert audio chunk into the processor
                    online_processor.insert_audio_chunk(audio_data)

                    # Process the audio chunk
                    result = online_processor.process_iter()
                    if result[2]:  # If there is transcribed text
                        start_time, end_time, text = result
                        output_time = time.time()

                        words = text.strip().split()
                        segment_duration = end_time - start_time
                        delay_per_word = segment_duration / len(words) if words else 0


                        transcription_history += " " + result[2]
                        print(f"Transcription: {transcription_history.strip()}")

                        # Perform vishing detection
                        # inputs = tokenizer(transcription_history, return_tensors="pt", truncation=True, max_length=512)
                        # outputs = model_inference(**inputs)
                        # predicted_class = outputs.logits.argmax(dim=-1).item()

                        # probabilities = F.softmax(outputs.logits, dim=-1)  # Apply softmax to logits
                        # vishing_probability = probabilities[0][1].item()  # Probability of the "Vishing" class
                        # not_vishing_probability = probabilities[0][0].item()
                        hasil = Detector.predict(transcription_history)
                        
                        text = hasil['text']
                        text_cleaned = hasil['cleaned']
                        label = hasil['predicted_label']
                        probability = hasil['probability']
                        
                        vishing_probability = probability * 100

                        if vishing_probability > threshold:
                            check_vishing = time.time()
                            prediction = 1
                                            
                            print("ALERT: High probability of Vishing detected!")
                        else:
                            prediction = 0
                            print("No alert: Probability below threshold.")
                        
                
        except KeyboardInterrupt:
            while not audio_queue.empty():
                print("processing last chunk")
                audio_chunk = audio_queue.get()
                audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
                online_processor.insert_audio_chunk(audio_data)

            flag_stop = True
            silence = np.zeros(int(1 * 16000), dtype=np.float32)  # 0.5 detik silence
            online_processor.insert_audio_chunk(silence)
            result = online_processor.process_iter()
            transcription_history += " " + result[2]

            final_result = online_processor.finish()
            if final_result[2]:
                start_time, end_time, text = final_result
                output_time = time.time()

                words = text.strip().split()
                segment_duration = end_time - start_time
                delay_per_word = segment_duration / len(words) if words else 0


                transcription_history += " " + final_result[2]
                hasil = Detector.predict(transcription_history)
                
                text = hasil['text']
                text_cleaned = hasil['cleaned']
                label = hasil['predicted_label']
                probability = hasil['probability']
                
                vishing_probability = probability * 100

                if vishing_probability > threshold:
                    check_vishing = time.time()
                    prediction = 1
                                    
                    print("ALERT: High probability of Vishing detected!")
                else:
                    prediction = 0
                    print("No alert: Probability below threshold.")
                print(f"Final Transcription: {transcription_history.strip()}")

    else:
        # File-based transcription and inference
        print(f"Starting offline transcription on {audio_file} using {model_type} ({model_size})")

        from faster_whisper import WhisperModel
        # Load model
        model = WhisperModel(
        model_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type="float16" if torch.cuda.is_available() else "int8"
        )

        # Transcribe
        segments, info = model.transcribe(audio_file, beam_size=5)
        transcription_history = ""

        audio_start_wall_time = time.time()  # Simulasi awal

        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            text = segment.text.strip()

            transcription_history += " " + text

            # Hitung delay per kata (simulasi)
            output_time = time.time()
            words = text.split()
            segment_duration = end_time - start_time
            delay_per_word = segment_duration / len(words) if words else 0



        print(f"Final Transcription: {transcription_history.strip()}")

        # Vishing detection
        hasil = Detector.predict(transcription_history.strip())
        text = hasil['text']
        text_cleaned = hasil['cleaned']
        label = hasil['predicted_label']
        probability = hasil['probability']

        vishing_probability = probability * 100
        if vishing_probability > 80:
            prediction = 1
            print("ALERT: High probability of Vishing detected!")
        else:
            prediction = 0
            print("No alert: Probability below threshold.")

        return transcription_history.strip()


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Transcribe audio and classify it as Vishing or Not Vishing.")
    parser.add_argument(
        "--audio_file", 
        type=str, 
        required=False, 
        help="Path to the audio file to be transcribed."
    )
    parser.add_argument(
        "--model_type", 
        type=str, 
        choices=["whisper", "faster_whisper"], 
        default="whisper", 
        help="Type of transcription model to use (default: whisper)."
    )
    parser.add_argument(
        "--model_size", 
        type=str, 
        choices=["tiny", "base", "small", "medium", "large"], 
        default="base", 
        help="Size of the Whisper model to use (default: base)."
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debugging to log performance metrics."
    )
    parser.add_argument(
        "--log_file", 
        type=str, 
        default="performance.log", 
        help="Path to the log file for performance metrics (default: performance.log)."
    )
    parser.add_argument(
        "--real_time", 
        action="store_true", 
        help="Enable real-time transcription mode."
    )

    # Parse arguments
    args = parser.parse_args()

    from argparse import Namespace

    # Simulate argparse arguments for asr_factory
    args_for_asr_factory = Namespace(
        lan="id",  # Language code for Indonesian
        model=args.model_size,
        backend="faster-whisper",
        model_cache_dir=None,
        model_dir=None,
        vad=False,
        vac=False,
        vac_chunk_size=0.04,
        min_chunk_size=0.8,  # Duration of each audio chunk in seconds
        buffer_trimming="segment",
        buffer_trimming_sec=5,
        log_level="INFO",
        task="transcribe",  # Add the task attribute
    )

    # Initialize ASR and Online Processor
    asr, online_processor = asr_factory(args_for_asr_factory)

    from final_model.vishing_detector import VishingDetector

    Detector = VishingDetector()

    # Call the main function with parsed arguments
    main(
        audio_file=args.audio_file, 
        model_type=args.model_type, 
        model_size=args.model_size, 
        debug=args.debug, 
        log_file=args.log_file,
        real_time=args.real_time
    )