import streamlit as st
import os
from subprocess import run,PIPE
import tempfile
import shutil
import subprocess

# title 
st.title("Video Stabilizer")
st.write("Upload a video, and we'll stabilize it for you!")

# video upload
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

# Process video if a file is uploaded
if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        input_video_path = temp_input_file.name

    # Create a temporary directory for output
    output_dir = tempfile.mkdtemp()
    col1, col2 = st.columns(2)
    
    
    # Display input video
    with col1:
        st.header("Input Video")
        st.video(input_video_path)
        print(input_video_path)

    # st.video(input_video_path)
    # print(input_video_path)

    # Button to process the video
    if st.button("Stabilize Video"):
        processing_msg=st.empty()
        processing_msg.write("Processing the video...")

        try:
            # Run the Python script to stabilize the video
            keyword = os.path.splitext(uploaded_file.name)[0]  
            result = run(
                ["python", "main.py", input_video_path, keyword],
                stdout=PIPE, stderr=PIPE, text=True
            )
            
            
            stabilized_video_path=f"C:/Users/Chitraksh/OneDrive/Desktop/DIP Project Final/DIP Project Final/output/{keyword}/stabilized_video.mp4"
            
            print(stabilized_video_path)

            # reencode the stablized video
            reencoded_path = stabilized_video_path.replace(".mp4", "_reencoded.mp4")
            subprocess.run(["ffmpeg", "-i", stabilized_video_path, "-vcodec", "libx264", "-crf", "23", "-preset", "fast", "-acodec", "aac", reencoded_path], check=True)
            print(reencoded_path)

            # Check if the output video was created
            if os.path.exists(reencoded_path):
                processing_msg.empty()
                st.success("Video stabilization complete!")

                # Display the stabilized video
                with col2:
                    st.header("Stabilized Video")
                    st.video(reencoded_path)
                    print("done")

                # st.video(reencoded_path)
                # print("done")

                # capturing calculated metrics from main.py
                filtered_output = "\n".join(
                    line for line in result.stdout.splitlines()
                    if "Loading Video..." in line or
                    "frames generated" in line or
                    "metrics" in line or
                    "Avg" in line
                )
                st.code(filtered_output)
            else:
                st.error("Error: Stabilized video not created.")
        except Exception as e:
            st.error(f"An error occurred while processing the video: {e}")
        finally:
            # Clean up temporary files
            os.remove(input_video_path)
            shutil.rmtree(output_dir)

