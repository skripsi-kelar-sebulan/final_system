# gradio_gui.py (Corrected Stop Button Logic)

import gradio as gr
from argparse import Namespace
import time
from main_app import VishingApp

# --- App Initialization ---
args = Namespace(model_size='small')
app = VishingApp(args)

# --- Gradio Functions ---

def realtime_vishing_detection(run_state):
    """
    Generator function for the REAL-TIME tab.
    """
    if run_state['running']: # Prevent starting if already running
        return

    run_state['running'] = True
    app.is_running = True
    label_output = None
    
    # Initial UI update when starting
    yield {
        transcription_textbox_realtime: "Starting microphone, please wait...",
        prediction_label_realtime: None,
        status_textbox: gr.update(value="Status: Listening..."),
        start_button: gr.update(interactive=False),
        stop_button: gr.update(interactive=True)
    }
    
    # Main processing loop
    for result_text in app.start_realtime_processing():
        if not run_state['running']:
            print("Stop flag detected, breaking loop.") # Debugging print
            break # Exit gracefully if stop button was pressed
        
        transcript = result_text['text']
        if result_text['label'] is not None:
            label = result_text['label']
        if result_text['score'] is not None:
            score = result_text['score']
        status = result_text['Status']

        if result_text['label'] is not None:
            if label == "Vishing":
                label_output = {"Vishing": score, "Not Vishing": 1-score}
            else: # Handles "Not Vishing" and any other label like "No Speech"
                label_output = {"Not Vishing": 1-score, "Vishing": score}
        
        if label_output is None:
            label_output = {"Not Vishing": "NaN", "Vishing": "NaN"}

        yield {
            transcription_textbox_realtime: gr.update(value=transcript),
            prediction_label_realtime: gr.update(value=label_output),
            status_textbox: gr.update(value=status),
        }
        time.sleep(0.1)

    # Final UI update after the loop has gracefully stopped
    print("Loop finished, sending final UI update.") # Debugging print
    yield {
        transcription_textbox_realtime: gr.update(value=app.transcription_history), # Show final transcript
        prediction_label_realtime: gr.update(),
        status_textbox: gr.update(value="Status: Stopped."),
        start_button: gr.update(interactive=True),
        stop_button: gr.update(interactive=False)
    }


def stop_detection(run_state):
    """
    -- CORRECTED LOGIC --
    Stops the real-time detection loop by setting flags and providing
    immediate UI feedback.
    """
    if run_state['running']:
        run_state['running'] = False
        app.stop_realtime_processing()
    
    # Return immediate feedback to the user
    return {
        status_textbox: gr.update(value="Status: Stopping..."),
        stop_button: gr.update(interactive=False) # Disable stop button immediately
    }


def offline_vishing_detection(audio_file):
    """
    Function for the OFFLINE tab. Processes an uploaded audio file.
    """
    if audio_file is None:
        return "Please upload an audio file first.", None, "Status: Waiting for file."

    result = app.process_offline(audio_file)
    label = result['label']
    score = result['score']
    
    if label == "Vishing":
        label_output = {"Vishing": score, "Not Vishing": 1-score}
    else: # Handles "Not Vishing" and any other label like "No Speech"
        label_output = {"Not Vishing": 1-score, "Vishing": score}
        
    return result['text'], label_output, "Status: Analysis Complete."


# --- Gradio UI Layout ---
# (The layout remains exactly the same as your previous version)
with gr.Blocks(theme=gr.themes.Soft(), title="Vishing Detection") as demo:
    run_state = gr.State(value={'running': False})

    gr.Markdown("# 🕵️ Automatic Vishing Detection System")
    
    with gr.Tabs():
        with gr.TabItem("🎙️ Real-time Detection"):
            gr.Markdown("Click 'Start Detection' to begin analyzing audio from your microphone in real-time.")
            with gr.Row():
                start_button = gr.Button("Start Detection", variant="primary")
                stop_button = gr.Button("Stop Detection", interactive=False)
            
            with gr.Row():
                with gr.Column():
                    transcription_textbox_realtime = gr.Textbox(lines=10, label="Live Transcription", interactive=False)
                with gr.Column():
                    prediction_label_realtime = gr.Label(label="Vishing Prediction")
                    status_textbox = gr.Textbox(label="Status", value="Status: Ready.", interactive=False)

        with gr.TabItem("📁 Offline File Analysis"):
            gr.Markdown("Upload an audio file (WAV, MP3) to get a full transcription and vishing prediction.")
            with gr.Row():
                audio_input_offline = gr.Audio(type="filepath", label="Upload Your Audio File")
            
            analyze_button_offline = gr.Button("Analyze Audio File", variant="primary")

            with gr.Row():
                 with gr.Column():
                    transcription_textbox_offline = gr.Textbox(lines=10, label="Full Transcription", interactive=False)
                 with gr.Column():
                    prediction_label_offline = gr.Label(label="Vishing Prediction")

    # --- Event Listeners ---
    
    # The outputs here describe what the GENERATOR will update over its lifetime
    start_event = start_button.click(
        fn=realtime_vishing_detection,
        inputs=[run_state],
        outputs=[transcription_textbox_realtime, prediction_label_realtime, status_textbox, start_button, stop_button]
    )
    
    # --- CORRECTED EVENT LISTENER ---
    stop_button.click(
        fn=stop_detection,
        inputs=[run_state],
        outputs=[status_textbox, stop_button], # The outputs of the stop_detection function
        cancels=None  # <-- This is the most important change: `cancels` is removed/set to None
    )

    analyze_button_offline.click(
        fn=offline_vishing_detection,
        inputs=[audio_input_offline],
        outputs=[transcription_textbox_offline, prediction_label_offline, status_textbox]
    )

# --- Launch the App ---
if __name__ == "__main__":
    demo.launch()