# main_app.py (Corrected for the "> 7 words" rule)

import argparse
from queue import Queue
import speech_recognition as sr
import numpy as np
import time
from final_model.vishing_detector import VishingDetector # Make sure this import is correct
from whisper_streaming.whisper_online import asr_factory # Make sure this import is correct
from faster_whisper import WhisperModel

class VishingApp:
    def __init__(self, args):
        self.args = args
        self.transcription_history = ""
        self.is_running = False
        print(f"Loading offline whisper model '{self.args.model_size}'...")
        self.offline_model = WhisperModel(self.args.model_size, device="cuda", compute_type="float16")

        # IMPORTANT: To run this code, either use the MockDetector for testing
        # or replace it with your actual VishingDetector and make sure you have the model files.
        self.detector = VishingDetector() # <-- UNCOMMENT THIS FOR YOUR REAL MODEL


    def _initialize_asr(self):
        from argparse import Namespace
        args_for_asr_factory = Namespace(lan="id", model=self.args.model_size, 
                                         backend="faster-whisper", 
                                         model_cache_dir=None, model_dir=None, 
                                         min_chunk_size=0.8, vac=False, 
                                         vac_chunk_size=0.04, buffer_trimming="segment", 
                                         buffer_trimming_sec=5, log_level="INFO", task="transcribe")
        asr, online_processor = asr_factory(args_for_asr_factory)
        return asr, online_processor

    def start_realtime_processing(self):
        """
        A generator that yields real-time updates. It will only start predicting
        after 7 or more words have been transcribed.
        """
        self.is_running = True
        self.transcription_history = ""
        asr, online_processor = self._initialize_asr()
        audio_queue = Queue()

        recorder = sr.Recognizer()
        recorder.energy_threshold = 1000 # Adjust as needed
        mic = sr.Microphone(sample_rate=16000)
        with mic:
            recorder.adjust_for_ambient_noise(mic)

        def record_callback(_, audio: sr.AudioData):
            audio_queue.put(audio.get_raw_data())

        listener = recorder.listen_in_background(mic, record_callback, phrase_time_limit=1)
        print("Real-time processing started. Listening...")

        while self.is_running:
            try:
                if not audio_queue.empty():
                    audio_chunk = audio_queue.get()
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0

                    online_processor.insert_audio_chunk(audio_data)
                    result = online_processor.process_iter()

                    if result and result[2]: # If there is new transcribed text
                        # Append new text to our history
                        self.transcription_history += " " + result[2]
                        transcript = self.transcription_history.strip()
                        
                        # =================================================================
                        # == THIS IS THE CORE LOGIC FOR YOUR REQUIREMENT ==
                        # =================================================================
                        word_count = len(transcript.split())

                        if word_count >= 7:
                            # If we have 7 or more words, perform prediction
                            prediction = self.detector.predict(transcript)
                            formatted_result = f"Label: {prediction['predicted_label']} (Score: {prediction['probability']:.2f})"
                            yield {"text": self.transcription_history,
                                    "label": prediction['predicted_label'],
                                    "score": prediction['probability'],
                                      "Status" : "Predicting"}
                        else:
                            # If we have fewer than 7 words, wait and do not predict
                            needed = 7 - word_count
                            yield {"text": self.transcription_history,
                                    "label": None,
                                    "score": None, 
                                    "Status" : f"Listening... ({needed} more words needed to predict)"}
                            
                        # =================================================================

                time.sleep(0.05)
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        listener()
        print("Real-time processing stopped.")
        yield self.transcription_history.strip(), "Processing stopped.", "Status: Stopped"

    def stop_realtime_processing(self):
        """Stops the real-time processing loop."""
        print("Stop signal received.")
        self.is_running = False

    def process_offline(self, audio_file_path):
        """
        --- REWRITTEN: This now uses faster-whisper for real transcription ---
        Handles transcription and detection for a given audio file.
        Returns a dictionary with the results.
        """
        print(f"Starting offline transcription on {audio_file_path}...")

        # 1. Transcribe the audio file using the pre-loaded model
        #    The 'language' parameter can be set to 'id' for Indonesian.
        segments, info = self.offline_model.transcribe(audio_file_path, language="id", beam_size=5)

        # 2. Join the transcribed segments into a single string
        full_transcription = " ".join(segment.text for segment in segments).strip()
        self.transcription_history = full_transcription
        print(f"Transcription complete: '{full_transcription}'")

        # 3. Get the prediction from the vishing detector
        prediction = self.detector.predict(self.transcription_history)
        print(prediction)

        # 4. Return the structured dictionary (the format is the same)
        return {
            "text": self.transcription_history,
            "label": prediction['predicted_label'],
            "score": prediction['probability']
        }